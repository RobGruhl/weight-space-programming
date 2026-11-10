"""Pure analysis helpers for the weights-reading validation (Claude). No model dependencies."""
import random

import torch


def m1_from_attn(attn, lo=16):
    """attn: heads x T x T probabilities. Mean over query positions p in [lo, T-1] of attn[:, p, p-1]."""
    t = attn.shape[-1]
    idx = torch.arange(lo, t)
    return attn[:, idx, idx - 1].mean(dim=1)


def profile_from_attn(attn, offsets=48):
    """Mean over query positions p in [offsets-1, T-1] of attn[:, p, p-d] for d in 0..offsets-1 -> heads x offsets."""
    t = attn.shape[-1]
    qs = torch.arange(offsets - 1, t)
    cols = qs[:, None] - torch.arange(offsets)[None, :]          # Q x offsets
    return attn[:, qs[:, None], cols].mean(dim=1)


def _ranks(values):
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        avg = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def spearman(x, y):
    rx, ry = _ranks(list(x)), _ranks(list(y))
    n = len(rx)
    mx, my = sum(rx) / n, sum(ry) / n
    cov = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    vx = sum((a - mx) ** 2 for a in rx) ** 0.5
    vy = sum((b - my) ** 2 for b in ry) ** 0.5
    return cov / (vx * vy) if vx and vy else 0.0


def permutation_p(x, y, n=10000, seed=20261107):
    """One-sided p-value for rho >= observed under random reassignment of x to the items of y."""
    observed = spearman(x, y)
    rng = random.Random(seed)
    x = list(x)
    count = 0
    for _ in range(n):
        rng.shuffle(x)
        if spearman(x, y) >= observed - 1e-12:
            count += 1
    return observed, (count + 1) / (n + 1)
