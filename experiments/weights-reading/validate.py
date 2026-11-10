"""Validation probe (Claude). Runs the unchanged Qwen3.5-0.8B on the frozen synthetic set, captures attention
probabilities of the six full-attention layers with forward hooks, and evaluates the frozen predictions.

    python validate.py --predictions-sha256 <hash of PREDICTIONS.json as logged> [--device mps]
"""
import os
os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', TOKENIZERS_PARALLELISM='false')
import argparse
import hashlib
import json
import platform
import resource
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
import transformers
from safetensors import safe_open
from transformers import Qwen3_5ForCausalLM, Qwen3_5TextConfig

import analysis

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
BASE = ROOT / 'private/tiny-replay/model'
OUT = ROOT / 'private/weights-reading'
MODEL_SHA = '04b1c301231dd422b8860db31311ab2721511346a32cb1e079c4c4e5f1fe4696'
FULL = [3, 7, 11, 15, 19, 23]


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 24), b''):
            h.update(chunk)
    return h.hexdigest()


def verdict(pred_key, supported, contradicted):
    return 'supported' if supported else ('contradicted' if contradicted else 'inconclusive')


def evaluate(pred, m1, label):
    names = [r['name'] for r in pred['ranking']]
    score = {r['name']: r['score'] for r in pred['ranking']}
    xs = [score[n] for n in names]
    ys = [m1[n] for n in names]
    rho, p = analysis.permutation_p(xs, ys, n=10000, seed=20261107)
    mean = lambda ns: sum(m1[n] for n in ns) / len(ns)
    top, comp, bot = mean(pred['top3']), mean(pred['comparators_median_trio_ranks_24_to_26']), mean(pred['bottom3'])
    l19, l3 = mean(pred['layer19_heads']), mean(pred['layer3_heads'])
    median = sorted(ys)[len(ys) // 2]
    components = {k: analysis.spearman([r[k] for r in pred['ranking']], ys) for k in ('contrast1', 'hf4_share', 'p1_v1')}
    return {'label': label,
            'P1_rank': {'rho': rho, 'permutation_p': p, 'verdict': verdict('P1', rho > 0.5 and p < 0.01, rho < 0.2)},
            'P2_top_vs_comparators': {'top3_mean_m1': top, 'median_trio_mean_m1': comp, 'bottom3_mean_m1': bot,
                                      'ratio_top_over_median_trio': top / comp if comp else None, 'ratio_top_over_bottom3': top / bot if bot else None,
                                      'verdict': verdict('P2', comp > 0 and bot > 0 and top >= 2 * comp and top >= 2 * bot, top <= comp)},
            'P3_layer19_nonlocal': {'layer19_mean_m1': l19, 'layer3_mean_m1': l3, 'median_head_m1': median,
                                    'verdict': verdict('P3', l19 < median and l19 <= 0.5 * l3, l19 >= l3)},
            'component_spearman': components,
            'max_m1': max(m1.items(), key=lambda kv: kv[1]),
            'per_layer_mean_m1': {f'L{L}': mean([n for n in names if n.startswith(f'L{L}.')]) for L in FULL}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--predictions-sha256', required=True)
    ap.add_argument('--device', default='mps')
    args = ap.parse_args()
    actual = sha256_file(HERE / 'PREDICTIONS.json')
    assert actual == args.predictions_sha256, f'PREDICTIONS.json hash {actual} does not match the logged hash'
    pred = json.loads((HERE / 'PREDICTIONS.json').read_text())
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    run = OUT / f'run-{stamp}'
    run.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    manifest = {'status': 'started', 'start_utc': stamp, 'predictions_sha256': actual, 'protocol_sha256': sha256_file(HERE / 'PROTOCOL.md'),
                'validate_sha256': sha256_file(HERE / 'validate.py'), 'analysis_sha256': sha256_file(HERE / 'analysis.py'),
                'model_file_sha256': sha256_file(BASE / 'model.safetensors'), 'device': args.device, 'attn_implementation': 'eager',
                'python': platform.python_version(), 'torch': torch.__version__, 'transformers': transformers.__version__}
    assert manifest['model_file_sha256'] == MODEL_SHA
    (run / 'manifest.json').write_text(json.dumps(manifest, indent=1) + '\n')
    torch.set_num_threads(4)
    config = Qwen3_5TextConfig(**json.loads((BASE / 'config.json').read_text())['text_config'])
    config._attn_implementation = 'eager'
    torch.set_default_dtype(torch.bfloat16)
    try:
        model = Qwen3_5ForCausalLM(config)
    finally:
        torch.set_default_dtype(torch.float32)
    with safe_open(BASE / 'model.safetensors', framework='pt', device='cpu') as f:
        state = {k.replace('model.language_model.', 'model.', 1): f.get_tensor(k) for k in f.keys() if k.startswith('model.language_model.')}
    if 'lm_head.weight' not in state:
        state['lm_head.weight'] = state['model.embed_tokens.weight']
    model.load_state_dict(state, strict=True, assign=True)
    del state
    model.eval().to(args.device)
    captured = {}

    def make_hook(layer):
        def hook(module, inputs, output):
            captured[layer] = output[1].detach().float().cpu()
        return hook
    handles = [model.model.layers[L].self_attn.register_forward_hook(make_hook(L)) for L in FULL]
    sets = {'A': pred['measurement']['sets']['A_random_tokens']['ids'], 'B': pred['measurement']['sets']['B_english']['ids_truncated_to_64']}
    per_seq = []
    for set_name, seqs in sets.items():
        for i, ids in enumerate(seqs):
            assert len(ids) == 64
            x = torch.tensor([ids], device=args.device)
            captured.clear()
            with torch.inference_mode():
                model.model(input_ids=x, use_cache=False)
            row = {'set': set_name, 'index': i, 'm1': {}, 'profile': {}, 'row_sum_max_abs_dev': 0.0}
            for L in FULL:
                attn = captured[L][0]
                assert attn.shape == (8, 64, 64), attn.shape
                row['row_sum_max_abs_dev'] = max(row['row_sum_max_abs_dev'], (attn.sum(-1) - 1).abs().max().item())
                m = analysis.m1_from_attn(attn, 16)
                prof = analysis.profile_from_attn(attn, 48)
                for h in range(8):
                    row['m1'][f'L{L}.H{h}'] = m[h].item()
                    row['profile'][f'L{L}.H{h}'] = [round(v, 6) for v in prof[h].tolist()]
            per_seq.append(row)
            print(json.dumps({'set': set_name, 'index': i, 'row_sum_dev': round(row['row_sum_max_abs_dev'], 4),
                              'top3_m1': {n: round(row['m1'][n], 3) for n in pred['top3']}}), flush=True)
    for h in handles:
        h.remove()
    (run / 'measured-per-sequence.json').write_text(json.dumps(per_seq, indent=1) + '\n')
    names = [r['name'] for r in pred['ranking']]

    def aggregate(rows):
        m1 = {n: sum(r['m1'][n] for r in rows) / len(rows) for n in names}
        prof = {n: [sum(r['profile'][n][d] for r in rows) / len(rows) for d in range(48)] for n in names}
        return m1, prof
    pooled_m1, pooled_prof = aggregate(per_seq)
    a_m1, _ = aggregate([r for r in per_seq if r['set'] == 'A'])
    b_m1, _ = aggregate([r for r in per_seq if r['set'] == 'B'])
    results = {'run': run.name, 'predictions_sha256': actual, 'model_file_sha256': MODEL_SHA,
               'primary_pooled': evaluate(pred, pooled_m1, 'pooled A+B'),
               'secondary_set_A_random': evaluate(pred, a_m1, 'set A random tokens'),
               'secondary_set_B_english': evaluate(pred, b_m1, 'set B English'),
               'measured_m1_pooled': {n: round(pooled_m1[n], 5) for n in names},
               'measured_argmax_offset_1_to_47_pooled': {n: int(max(range(1, 48), key=lambda d: pooled_prof[n][d])) for n in names},
               'measured_mean_profile_pooled': {n: [round(v, 5) for v in pooled_prof[n]] for n in names},
               'max_row_sum_abs_dev': max(r['row_sum_max_abs_dev'] for r in per_seq),
               'seconds': round(time.monotonic() - start, 1), 'peak_rss_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
    (run / 'results.json').write_text(json.dumps(results, indent=1) + '\n')
    (HERE / 'RESULTS.json').write_text(json.dumps(results, indent=1) + '\n')
    manifest.update(status='complete', seconds=results['seconds'], peak_rss_bytes=results['peak_rss_bytes'])
    (run / 'manifest.json').write_text(json.dumps(manifest, indent=1) + '\n')
    prim = results['primary_pooled']
    print(json.dumps({k: prim[k] for k in ('P1_rank', 'P2_top_vs_comparators', 'P3_layer19_nonlocal', 'component_spearman', 'max_m1', 'per_layer_mean_m1')}, indent=1))
    print('seconds', results['seconds'])


if __name__ == '__main__':
    main()
