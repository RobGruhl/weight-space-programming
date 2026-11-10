"""Correctness tests for the weights-only reader against the installed Transformers Qwen3.5 attention code.

These tests use random tensors and a tiny randomly initialized attention module; no checkpoint, no data.
"""
import unittest
import torch
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5Attention, Qwen3_5TextRotaryEmbedding, apply_rotary_pos_emb
import wsr


class ReaderTests(unittest.TestCase):
    def test_rotary_scores_match_explicit_rotation(self):
        torch.manual_seed(0)
        d, rot = 32, 8
        inv = wsr.inv_freq(rot, 1e4)
        q, k = torch.randn(5, d), torch.randn(7, d)
        offsets = torch.arange(0, 6)
        closed = wsr.rotary_scores(q, k, offsets, inv, rot, 1.0)
        for oi, off in enumerate(offsets):
            qr = wsr.rotate(q, torch.full((5,), int(off)), inv, rot)
            kr = wsr.rotate(k, torch.zeros(7, dtype=torch.long), inv, rot)
            self.assertTrue(torch.allclose(closed[:, :, oi], qr @ kr.T, atol=1e-5))

    def test_rotate_matches_hf_apply_rotary(self):
        torch.manual_seed(1)
        rot, d, t = 8, 32, 6
        inv = wsr.inv_freq(rot, 1e4)
        pos = torch.arange(t)
        freqs = pos[:, None].float() * inv[None, :]
        cos, sin = torch.cat([freqs, freqs], -1).cos()[None], torch.cat([freqs, freqs], -1).sin()[None]
        q = torch.randn(1, 1, t, d)
        k = torch.randn(1, 1, t, d)
        qe, ke = apply_rotary_pos_emb(q, k, cos, sin)
        self.assertTrue(torch.allclose(qe[0, 0], wsr.rotate(q[0, 0], pos, inv, rot), atol=1e-6))
        self.assertTrue(torch.allclose(ke[0, 0], wsr.rotate(k[0, 0], pos, inv, rot), atol=1e-6))

    def test_reader_matches_tiny_hf_attention_module(self):
        torch.manual_seed(2)
        cfg = Qwen3_5TextConfig(hidden_size=32, num_attention_heads=4, num_key_value_heads=2, head_dim=16,
                                num_hidden_layers=1, layer_types=['full_attention'], attention_bias=False,
                                rms_norm_eps=1e-6, rope_parameters={'rope_type': 'default', 'rope_theta': 1e7,
                                                                     'partial_rotary_factor': 0.25,
                                                                     'mrope_interleaved': True, 'mrope_section': [1, 1, 0]},
                                vocab_size=64, intermediate_size=64)
        cfg._attn_implementation = 'eager'
        attn = Qwen3_5Attention(cfg, 0).eval()
        with torch.no_grad():
            for p in attn.parameters():
                p.copy_(torch.randn_like(p) * 0.3)
        rope = Qwen3_5TextRotaryEmbedding(cfg)
        t = 9
        x = torch.randn(1, t, 32)
        pos = torch.arange(t).view(1, 1, t).expand(3, 1, t)
        cos, sin = rope(x, pos)
        mask = torch.full((t, t), float('-inf')).triu(1)[None, None]
        # The decoder layer applies input_layernorm before self_attn, so the reader normalizes x itself (zero gain
        # weight here) and the library module is fed the same normalized states.
        xn = wsr.rmsnorm(x[0], torch.zeros(32))
        with torch.no_grad():
            mine = wsr.attention_probs_from_weights(x[0], torch.zeros(32), attn.q_proj.weight, attn.k_proj.weight,
                                                   attn.q_norm.weight, attn.k_norm.weight, 4, 2, 16, 4, 1e7)
            _, hf_probs = attn(xn[None], (cos, sin), mask)
        self.assertEqual(hf_probs.shape, (1, 4, t, t))
        self.assertTrue(torch.allclose(hf_probs[0], mine, atol=1e-5), (hf_probs[0] - mine).abs().max())


if __name__ == '__main__':
    unittest.main()
