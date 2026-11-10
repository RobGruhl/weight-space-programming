"""Freeze the weights-reading predictions from the two weights-only signals (Claude). No activations, no text.

Signals per head (both computed before this script, from weights only):
  contrast1: stand-in mean score at offset 1 minus the mean over offsets 2..47 (from the v1 pass);
  hf4: share of the head's rotary query-key energy in the four highest rotary frequencies, embedding-second-moment
       input assumption (from diagnose_weights.py).
Predicted previous-token score = negative mean of the two ranks (1 = largest), tie-break on contrast1.
"""
import os
os.environ.update(TOKENIZERS_PARALLELISM='false', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
import json
from datetime import datetime, timezone
from pathlib import Path

import torch

from read_weights import MODEL_DIR, SEED_SET_A, SET_B_TEXT, TOKEN_HI, TOKEN_LO, sha256_file

HERE = Path(__file__).resolve().parent
profiles = json.loads((HERE / 'predicted-profiles-v1-flat.json').read_text())
diag = json.loads((HERE / 'weights-diagnostics.json').read_text())
p1 = {h['name']: h['p1'] for h in profiles['heads']}
contrast = {r['name']: r['mean_score_contrast1'] for r in diag['rows']}
hf4 = {r['name']: r['hf4_total_emp'] for r in diag['rows']}
names = sorted(p1)
assert len(names) == 48 and set(names) == set(contrast) == set(hf4)


def ranks(d):
    order = sorted(names, key=lambda n: -d[n])
    return {n: i + 1 for i, n in enumerate(order)}


r_c, r_h = ranks(contrast), ranks(hf4)
score = {n: -(r_c[n] + r_h[n]) / 2 for n in names}
ranked = sorted(names, key=lambda n: (-score[n], -contrast[n]))
top3, comparators, bottom3 = ranked[:3], ranked[23:26], ranked[-3:]
layer = lambda L: [n for n in names if n.startswith(f'L{L}.')]

gen_a = torch.Generator().manual_seed(SEED_SET_A)
set_a = torch.randint(TOKEN_LO, TOKEN_HI, (10, 64), generator=gen_a).tolist()
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(MODEL_DIR, local_files_only=True, trust_remote_code=False)
set_b = []
for text in SET_B_TEXT:
    enc = tok.encode(text, add_special_tokens=False)
    assert len(enc) >= 64, len(enc)
    set_b.append(enc[:64])

predictions = {
    'frozen_utc': datetime.now(timezone.utc).isoformat(),
    'reader': 'the assistant in this session (Claude Fable 5.1); not the published editor model, no identity continuity claimed',
    'inputs': {'predicted_profiles_v1_sha256': sha256_file(HERE / 'predicted-profiles-v1-flat.json'),
               'weights_diagnostics_sha256': sha256_file(HERE / 'weights-diagnostics.json'),
               'protocol_sha256': sha256_file(HERE / 'PROTOCOL.md')},
    'reading_summary': [
        'v1 stand-in pass: predicted mass at offset 1 is 0.019 to 0.027 for all heads (uniform would be 0.021); embedding-row stand-ins carry almost no content-independent positional signal.',
        'No head puts more than the uniform 25% share of query-key energy into rotary dimensions; no head is purely positional.',
        'All eight layer-19 heads put under 1.5% of their rotary energy into the four highest frequencies (uniform 12.5%); they cannot resolve offsets of one to three tokens.',
        'Layer 3 and layer 7 heads retain the most high-frequency rotary energy; L3.H1 and L3.H2 also show the largest stand-in offset-1 contrast.',
        'The pair-level fraction preferring offset 1 is an endpoint artifact of slowly varying low frequencies and is not used.'],
    'ranking': [{'name': n, 'score': score[n], 'rank_contrast1': r_c[n], 'rank_hf4': r_h[n], 'contrast1': round(contrast[n], 5),
                 'hf4_share': round(hf4[n], 5), 'p1_v1': round(p1[n], 5)} for n in ranked],
    'top3': top3, 'comparators_median_trio_ranks_24_to_26': comparators, 'bottom3': bottom3,
    'layer19_heads': layer(19), 'layer3_heads': layer(3),
    'predictions': {
        'P1_rank': {'statement': 'Spearman rank correlation between the frozen score and measured m1 across all 48 heads, pooled set A+B.',
                    'supported_if': 'rho > 0.5 and one-sided permutation p < 0.01 (10,000 permutations, seed 20261107)',
                    'contradicted_if': 'rho < 0.2', 'otherwise': 'inconclusive'},
        'P2_top_vs_comparators': {'statement': 'Mean measured m1 of the top-3 heads is at least 2x the mean m1 of the median trio and at least 2x the mean m1 of the bottom-3, pooled.',
                                  'supported_if': 'both ratios >= 2', 'contradicted_if': 'top-3 mean m1 <= median-trio mean m1', 'otherwise': 'inconclusive'},
        'P3_layer19_nonlocal': {'statement': 'Mean m1 over the eight layer-19 heads is below the median head m1 and at most 0.5x the mean m1 of the eight layer-3 heads, pooled.',
                                'supported_if': 'both hold', 'contradicted_if': 'layer-19 mean m1 >= layer-3 mean m1', 'otherwise': 'inconclusive'}},
    'secondary_descriptive': ['Spearman of each component alone (contrast1, hf4, p1_v1) with m1', 'set A and set B separately for P1-P3',
                              'maximum m1 and which head; the stand-in reading cannot predict its absolute level', 'argmax offset 1..47 of the mean profile for every head',
                              'per-layer mean m1 and mean profile'],
    'measurement': {'model': 'unchanged Qwen3.5-0.8B, eager attention, forward hooks on the six self_attn modules, no sampling',
                    'sequence_length': 64,
                    'sets': {'A_random_tokens': {'seed': SEED_SET_A, 'token_id_range': [TOKEN_LO, TOKEN_HI], 'ids': set_a},
                             'B_english': {'texts': SET_B_TEXT, 'ids_truncated_to_64': set_b}},
                    'm1': 'mean over sequences and query positions 16..63 of attention[h, p, p-1]',
                    'profile': 'mean over sequences and query positions 47..63 of attention[h, p, p-d], d in 0..47'},
    'controls': ['set A removes lexical content as an explanation', 'set B checks survival on natural text', 'comparators preselected by rank',
                 'permutation null preregistered', 'bottom-3 and layer-19 are negative predictions'],
    'not_claimed': ['causality', 'semantic meaning of any head', 'generalization beyond this model and these sequences'],
}
(HERE / 'PREDICTIONS.json').write_text(json.dumps(predictions, indent=1) + '\n')
print('top3', top3, '| comparators', comparators, '| bottom3', bottom3)
print('component ranks top3:', [(n, r_c[n], r_h[n]) for n in top3])
print('PREDICTIONS.json sha256', sha256_file(HERE / 'PREDICTIONS.json'))
