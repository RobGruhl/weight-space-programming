"""Weights-only reading helpers for Qwen3.5 full-attention layers (Claude).

Reimplements the query/key path of `Qwen3_5Attention` from the installed Transformers code so that per-head
positional preferences can be computed from weights and stand-in input vectors, with no model forward pass.
"""
import torch

EPS = 1e-6


def rmsnorm(x, weight, eps=EPS):
    """Qwen3_5RMSNorm: x * rsqrt(mean(x^2) + eps) * (1 + weight)."""
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * (1.0 + weight)


def inv_freq(rot_dim, theta):
    return 1.0 / (theta ** (torch.arange(0, rot_dim, 2, dtype=torch.float32) / rot_dim))


def qk_from_weights(x_normed, wq, wk, gq, gk, heads, kv_heads, head_dim):
    """Query and key heads after per-head RMSNorm, before rotary. x_normed: N x hidden (already input-normed).

    q_proj rows are laid out per head as [query(head_dim) | gate(head_dim)]; `torch.chunk(..., 2, dim=-1)` on a
    view (N, heads, 2*head_dim) takes the first half as the query.
    """
    q = (x_normed @ wq.T).view(-1, heads, 2 * head_dim)[:, :, :head_dim]
    k = (x_normed @ wk.T).view(-1, kv_heads, head_dim)
    return rmsnorm(q, gq), rmsnorm(k, gk)


def rotate(v, positions, inv, rot_dim):
    """Rotary embedding in the HF `rotate_half` layout: pairs (d, d + rot_dim/2) share frequency inv[d]."""
    half = rot_dim // 2
    ang = positions[:, None].float() * inv[None, :]
    c, s = ang.cos(), ang.sin()
    v1, v2 = v[..., :half], v[..., half:rot_dim]
    return torch.cat([v1 * c - v2 * s, v2 * c + v1 * s, v[..., rot_dim:]], dim=-1)


def rotary_scores(qh, kh, offsets, inv, rot_dim, scaling):
    """Scaled scores for every (query, key) pair at every relative offset, query at position d and key at 0.

    qh: Nq x D, kh: Nk x D, offsets: O ints. Returns Nq x Nk x O. Uses the closed form
    Re(z_q conj(z_k) e^{i theta d}) = A cos(theta d) - B sin(theta d) per frequency, plus the non-rotary dot.
    """
    half = rot_dim // 2
    q1, q2 = qh[:, :half], qh[:, half:rot_dim]
    k1, k2 = kh[:, :half], kh[:, half:rot_dim]
    a = q1[:, None, :] * k1[None, :, :] + q2[:, None, :] * k2[None, :, :]
    b = q2[:, None, :] * k1[None, :, :] - q1[:, None, :] * k2[None, :, :]
    c = qh[:, rot_dim:] @ kh[:, rot_dim:].T
    ang = offsets[:, None].float() * inv[None, :]
    scores = a @ ang.cos().T - b @ ang.sin().T + c[..., None]
    return scores * scaling


def attention_probs_from_weights(x, w_in, wq, wk, gq, gk, heads, kv_heads, head_dim, rot_dim, theta):
    """Full causal attention probabilities (heads x T x T) for one sequence of residual inputs x (T x hidden)."""
    xn = rmsnorm(x, w_in)
    q, k = qk_from_weights(xn, wq, wk, gq, gk, heads, kv_heads, head_dim)
    t = x.shape[0]
    pos = torch.arange(t)
    inv = inv_freq(rot_dim, theta)
    groups = heads // kv_heads
    probs = []
    for h in range(heads):
        qr = rotate(q[:, h], pos, inv, rot_dim)
        kr = rotate(k[:, h // groups], pos, inv, rot_dim)
        s = (qr @ kr.T) * head_dim ** -0.5
        mask = torch.triu(torch.ones(t, t, dtype=torch.bool), diagonal=1)
        s = s.masked_fill(mask, float('-inf'))
        probs.append(torch.softmax(s, dim=-1))
    return torch.stack(probs)
