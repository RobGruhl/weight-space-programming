"""Weights-only diagnostics per full-attention head (Claude). No forward pass, no text.

For each head: (1) energy per query/key dimension after the per-head norm gain, under two input assumptions
(isotropic, and the embedding second-moment matrix from 4096 embedding rows); (2) the share of that energy in
rotary dims and in the four highest rotary frequencies; (3) from the stand-in pairs, the fraction of pairs whose
best offset in 1..47 is exactly 1, and the mean-score contrast at offset 1.
"""
import os
os.environ.update(OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', VECLIB_MAXIMUM_THREADS='2')
import json
from pathlib import Path
import torch
from safetensors import safe_open
import wsr

torch.set_num_threads(2)
ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
MODEL = ROOT / 'private/tiny-replay/model/model.safetensors'
P = 'model.language_model.'
FULL = [3, 7, 11, 15, 19, 23]
H, KV, D, ROT, THETA = 8, 2, 256, 64, 1e7
inv = wsr.inv_freq(ROT, THETA)
rows = []
gen = torch.Generator().manual_seed(20261107)
ids = torch.randint(1000, 60000, (384,), generator=gen)
gen2 = torch.Generator().manual_seed(20261109)
ids_cov = torch.randint(1000, 60000, (4096,), generator=gen2)
offsets = torch.arange(48)
with safe_open(MODEL, framework='pt', device='cpu') as f:
    emb = f.get_tensor(P + 'embed_tokens.weight')
    xq, xk = emb[ids[:192]].float(), emb[ids[192:]].float()
    xcov = emb[ids_cov].float()
    del emb
    for L in FULL:
        pre = f'{P}layers.{L}.'
        w_in = f.get_tensor(pre + 'input_layernorm.weight').float()
        wq = f.get_tensor(pre + 'self_attn.q_proj.weight').float()
        wk = f.get_tensor(pre + 'self_attn.k_proj.weight').float()
        gq = f.get_tensor(pre + 'self_attn.q_norm.weight').float()
        gk = f.get_tensor(pre + 'self_attn.k_norm.weight').float()
        xn_cov = wsr.rmsnorm(xcov, w_in)
        sigma = xn_cov.T @ xn_cov / xn_cov.shape[0]          # uncentered second moment, 1024 x 1024
        q, _ = wsr.qk_from_weights(wsr.rmsnorm(xq, w_in), wq, wk, gq, gk, H, KV, D)
        _, k = wsr.qk_from_weights(wsr.rmsnorm(xk, w_in), wq, wk, gq, gk, H, KV, D)
        for h in range(H):
            wq_h = wq.view(H, 2 * D, -1)[h, :D]                 # 256 x 1024 query rows
            wk_h = wk.view(KV, D, -1)[h // (H // KV)]
            def energies(w, g):
                iso = (1 + g) ** 2 * (w ** 2).sum(1)
                emp = (1 + g) ** 2 * ((w @ sigma) * w).sum(1)
                return iso, emp
            eq_iso, eq_emp = energies(wq_h, gq)
            ek_iso, ek_emp = energies(wk_h, gk)
            def shares(eq, ek):
                pf = torch.sqrt((eq[:32] + eq[32:64]) * (ek[:32] + ek[32:64]))     # per-frequency product, 32
                total = torch.sqrt(eq.sum() * ek.sum())
                return {'rot_share': (pf.sum() / total).item(), 'hf4_share_of_total': (pf[:4].sum() / total).item(),
                        'hf4_share_of_rot': (pf[:4].sum() / pf.sum()).item(), 'spectrum': [round(v, 4) for v in (pf / pf.sum()).tolist()]}
            s_iso, s_emp = shares(eq_iso, ek_iso), shares(eq_emp, ek_emp)
            scores = wsr.rotary_scores(q[:, h], k[:, h // (H // KV)], offsets, inv, ROT, D ** -0.5)
            arg = scores[:, :, 1:].argmax(-1) + 1
            frac1 = (arg == 1).float().mean().item()
            ms = scores.mean((0, 1))
            contrast1 = (ms[1] - ms[2:].mean()).item()
            rows.append({'name': f'L{L}.H{h}', 'rot_share_iso': s_iso['rot_share'], 'hf4_total_iso': s_iso['hf4_share_of_total'],
                         'rot_share_emp': s_emp['rot_share'], 'hf4_total_emp': s_emp['hf4_share_of_total'], 'hf4_of_rot_emp': s_emp['hf4_share_of_rot'],
                         'pair_frac_argmax1': frac1, 'mean_score_contrast1': contrast1, 'spectrum_emp': s_emp['spectrum']})
rows_sorted = sorted(rows, key=lambda r: -r['hf4_total_emp'])
print(f"{'head':9s} {'rot_iso':>8s} {'hf4_iso':>8s} {'rot_emp':>8s} {'hf4_emp':>8s} {'hf4/rot':>8s} {'frac@1':>7s} {'contr1':>7s}")
for r in rows_sorted:
    print(f"{r['name']:9s} {r['rot_share_iso']:8.3f} {r['hf4_total_iso']:8.3f} {r['rot_share_emp']:8.3f} {r['hf4_total_emp']:8.3f} {r['hf4_of_rot_emp']:8.3f} {r['pair_frac_argmax1']:7.3f} {r['mean_score_contrast1']:7.3f}")
(HERE / 'weights-diagnostics.json').write_text(json.dumps({'rows': rows, 'note': 'weights-only; isotropic and embedding-second-moment input assumptions; stand-in pair statistics'}, indent=1) + '\n')
