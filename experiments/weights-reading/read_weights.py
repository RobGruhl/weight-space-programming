"""Weights-only reading phase (Claude). No model forward pass, no activations from text, no labels.

For each of the 48 full-attention heads of Qwen3.5-0.8B, computes a predicted positional profile: the average
softmax attention mass a query would place on a key at relative offset d (0..47) when query and key come from
unrelated stand-in inputs. Stand-in inputs are rows of the embedding matrix (weights) for randomly drawn ordinary
token ids, passed through the layer's input RMSNorm and the head's query/key projections and norms exactly as the
installed modeling code does. Writes `predicted-profiles.json` and the frozen `PREDICTIONS.json`, which also fixes
the synthetic validation set. CPU only, two threads.
"""
import os
os.environ.update(OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', VECLIB_MAXIMUM_THREADS='2',
                  TOKENIZERS_PARALLELISM='false', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
import hashlib
import json
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
from safetensors import safe_open

import wsr

torch.set_num_threads(2)
ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
MODEL_DIR = ROOT / 'private/tiny-replay/model'
MODEL = MODEL_DIR / 'model.safetensors'
MODEL_SHA = '04b1c301231dd422b8860db31311ab2721511346a32cb1e079c4c4e5f1fe4696'
P = 'model.language_model.'
FULL_LAYERS = [3, 7, 11, 15, 19, 23]
HEADS, KV_HEADS, HEAD_DIM, ROT_DIM, THETA = 8, 2, 256, 64, 1e7
OFFSETS = 48
NQ = NK = 192
TOKEN_LO, TOKEN_HI = 1000, 60000
SEED_READ, SEED_SET_A, SEED_PERM = 20261107, 20261108, 20261107
SET_B_TEXT = [
    "The committee will meet on Thursday afternoon to review the three remaining proposals. Each presenter has fifteen minutes, followed by questions from the panel. Please bring printed copies of your budget tables, since the projector in that room has been unreliable all month. Lunch will be provided in the lobby beforehand, and parking validation is available at the front desk for anyone who drives in.",
    "To make the sauce, melt two tablespoons of butter in a wide pan over medium heat until it foams. Add the chopped shallots and cook them gently until they turn translucent, about four minutes. Pour in the white wine and let it reduce by half, then whisk in the cream and a pinch of salt. Simmer until the sauce coats the back of a spoon, and finish with lemon.",
    "A cold front will move across the region overnight, bringing scattered showers and gusty winds to the coast by early morning. Temperatures will fall into the low forties inland, with patchy frost possible in sheltered valleys. Skies should clear by midday on Saturday, and the weekend looks dry and bright, although a stiff northwesterly breeze will keep it feeling cooler than the numbers suggest.",
    "The function reads each record from the input stream, validates the header fields against the schema, and appends well-formed rows to the output buffer. Malformed rows are counted and logged with their line number but never raise an exception, because the caller expects partial results on bad input. When the buffer reaches the configured size, it is flushed to disk and cleared before processing continues.",
    "She had not expected the station to be so quiet at that hour. The departures board flickered between two delayed trains, and a single vendor was closing the shutters on the coffee kiosk. She found a bench near the far platform, set her bag down carefully, and watched the rain streak the glass roof while she waited for a message that might not come.",
    "This lightweight jacket is made from a recycled nylon shell with a water-resistant coating and a soft fleece lining at the collar. It packs down into its own front pocket, weighs under three hundred grams, and comes in four colors. The adjustable hood and elastic cuffs keep out drafts, while two zippered hand pockets hold a phone, keys, and a small wallet securely.",
    "To find the area of a triangle when you know two sides and the angle between them, multiply the two side lengths together, multiply by the sine of the included angle, and divide the result by two. This works because the height of the triangle equals one side times the sine of that angle, so the familiar formula of one half base times height reappears in disguise.",
    "The old town sits on a bend in the river, its narrow streets climbing from the quay toward a square shaded by plane trees. Most visitors arrive by the morning ferry, wander the market until the stalls close at noon, and then eat grilled fish at one of the terraces overlooking the water. In the evening the bells ring from the hilltop chapel across the valley.",
    "Thank you for sending the revised draft so quickly. I have read through the second and third sections and left a few comments in the margins, mostly about terminology and one place where the timeline seems to contradict the summary. Nothing major, but it would be worth a careful pass before it goes to the wider group next week. Let me know if you want to talk any of it through.",
    "Researchers measured the growth of the moss samples under three light conditions over eight weeks, recording mass and surface area every Monday. Samples kept in indirect light grew steadily, those in direct sunlight dried at the edges by week five, and the shaded group barely changed. The team cautions that the trays were watered by hand, so moisture levels were not perfectly controlled across groups.",
]


def sha256_bytes(b):
    return hashlib.sha256(b).hexdigest()


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 24), b''):
            h.update(chunk)
    return h.hexdigest()


def tensor_sha(t):
    return sha256_bytes(t.contiguous().view(torch.uint16).numpy().tobytes())


def head_name(layer, head):
    return f'L{layer}.H{head}'


def main():
    start = time.monotonic()
    model_sha = sha256_file(MODEL)
    assert model_sha == MODEL_SHA, 'base model file hash mismatch'
    gen = torch.Generator().manual_seed(SEED_READ)
    ids = torch.randint(TOKEN_LO, TOKEN_HI, (NQ + NK,), generator=gen)
    inv = wsr.inv_freq(ROT_DIM, THETA)
    offsets = torch.arange(OFFSETS)
    scaling = HEAD_DIM ** -0.5
    read_tensors = {}
    heads = []
    with safe_open(MODEL, framework='pt', device='cpu') as f:
        emb = f.get_tensor(P + 'embed_tokens.weight')
        read_tensors[P + 'embed_tokens.weight'] = {'shape': list(emb.shape), 'sha256': tensor_sha(emb), 'rows_used': 'ids below'}
        x_q = emb[ids[:NQ]].float()
        x_k = emb[ids[NQ:]].float()
        del emb
        for layer in FULL_LAYERS:
            pre = f'{P}layers.{layer}.'
            names = {'w_in': pre + 'input_layernorm.weight', 'wq': pre + 'self_attn.q_proj.weight', 'wk': pre + 'self_attn.k_proj.weight',
                     'gq': pre + 'self_attn.q_norm.weight', 'gk': pre + 'self_attn.k_norm.weight'}
            t = {}
            for key, name in names.items():
                raw = f.get_tensor(name)
                read_tensors[name] = {'shape': list(raw.shape), 'sha256': tensor_sha(raw)}
                t[key] = raw.float()
            q, _ = wsr.qk_from_weights(wsr.rmsnorm(x_q, t['w_in']), t['wq'], t['wk'], t['gq'], t['gk'], HEADS, KV_HEADS, HEAD_DIM)
            _, k = wsr.qk_from_weights(wsr.rmsnorm(x_k, t['w_in']), t['wq'], t['wk'], t['gq'], t['gk'], HEADS, KV_HEADS, HEAD_DIM)
            for h in range(HEADS):
                scores = wsr.rotary_scores(q[:, h], k[:, h // (HEADS // KV_HEADS)], offsets, inv, ROT_DIM, scaling)
                mass = torch.softmax(scores, dim=-1)
                profile = mass.mean(dim=(0, 1))
                mean_score = scores.mean(dim=(0, 1))
                # share of score variance across (pair, offset) that is explained by offset alone
                total_var = scores.var().item()
                offset_var = mean_score.var().item()
                heads.append({'name': head_name(layer, h), 'layer': layer, 'head': h, 'kv_head': h // (HEADS // KV_HEADS),
                              'p1': profile[1].item(), 'p0': profile[0].item(), 'p2': profile[2].item(),
                              'predicted_argmax_offset_1_to_47': int(profile[1:].argmax().item()) + 1,
                              'positional_variance_share': offset_var / total_var if total_var else None,
                              'profile_mass': [round(v, 6) for v in profile.tolist()],
                              'profile_mean_score': [round(v, 5) for v in mean_score.tolist()]})
            print(f'layer {layer}: p1 by head', [round(x['p1'], 3) for x in heads[-HEADS:]], flush=True)
    profiles = {'reader': 'the assistant in this session (Claude Fable 5.1), not the published editor model',
                'method': 'closed-form rotary scores between unrelated stand-in query and key inputs; softmax over offsets 0..47; mean mass per offset',
                'stand_in_inputs': {'source': 'rows of embed_tokens.weight', 'seed': SEED_READ, 'token_id_range': [TOKEN_LO, TOKEN_HI],
                                    'n_query_rows': NQ, 'n_key_rows': NK, 'ids': ids.tolist()},
                'architecture': {'full_attention_layers': FULL_LAYERS, 'heads': HEADS, 'kv_heads': KV_HEADS, 'head_dim': HEAD_DIM,
                                 'rotary_dim': ROT_DIM, 'rope_theta': THETA, 'scaling': scaling, 'offsets': OFFSETS},
                'model_file_sha256': model_sha, 'tensors_read': read_tensors, 'activations_used': False, 'text_inputs_used': False,
                'runtime': {'python': platform.python_version(), 'torch': torch.__version__, 'threads': 2, 'seconds': round(time.monotonic() - start, 2)},
                'heads': heads}
    (HERE / 'predicted-profiles.json').write_text(json.dumps(profiles, indent=1) + '\n')
    ranked = sorted(heads, key=lambda x: -x['p1'])
    names = [x['name'] for x in ranked]
    top3, comparators, bottom3 = names[:3], names[23:26], names[-3:]
    # validation set, fixed now, before any probe
    gen_a = torch.Generator().manual_seed(SEED_SET_A)
    set_a = torch.randint(TOKEN_LO, TOKEN_HI, (10, 64), generator=gen_a).tolist()
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL_DIR, local_files_only=True, trust_remote_code=False)
    set_b = []
    for text in SET_B_TEXT:
        enc = tok.encode(text, add_special_tokens=False)
        assert len(enc) >= 64, f'passage too short: {len(enc)} tokens'
        set_b.append(enc[:64])
    predictions = {
        'frozen_utc': datetime.now(timezone.utc).isoformat(),
        'reader': profiles['reader'],
        'predicted_profiles_sha256': sha256_file(HERE / 'predicted-profiles.json'),
        'ranking_by_predicted_p1': [{'name': x['name'], 'p1': round(x['p1'], 5), 'argmax_offset': x['predicted_argmax_offset_1_to_47']} for x in ranked],
        'top3_predicted_previous_token_heads': top3,
        'comparators_median_trio_ranks_24_to_26_of_48': comparators,
        'bottom3_predicted': bottom3,
        'predicted_argmax_offset_top3': {x['name']: x['predicted_argmax_offset_1_to_47'] for x in ranked[:3]},
        'predictions': {
            'P1_rank': {'statement': 'Across all 48 full-attention heads, Spearman rank correlation between predicted p1 (mass at offset 1 from weights) and measured m1 (mean attention at offset -1) on the pooled synthetic set.',
                        'supported_if': 'rho > 0.5 and permutation p < 0.01', 'contradicted_if': 'rho < 0.2', 'otherwise': 'inconclusive'},
            'P2_top_vs_comparators': {'statement': 'Mean measured m1 of the three top-ranked heads is at least 0.40 and at least 3x the mean m1 of the three median-ranked comparator heads, pooled set.',
                                      'supported_if': 'both conditions hold', 'contradicted_if': 'top3 mean m1 <= comparator mean m1', 'otherwise': 'inconclusive'},
            'P3_preferred_offset': {'statement': 'For each top-3 head, the measured argmax over offsets 1..47 of the mean attention profile equals the predicted argmax listed above.',
                                    'supported_if': 'at least 2 of 3 match', 'contradicted_if': '0 of 3 match', 'otherwise': 'inconclusive'}},
        'secondary_reports': ['same statistics on set A (random tokens) and set B (English) separately', 'bottom-3 m1 versus overall median m1', 'per-layer summary'],
        'measurement': {'model': 'unchanged Qwen3.5-0.8B, eager attention, MPS, no sampling', 'sequence_length': 64,
                        'sets': {'A_random_tokens': {'seed': SEED_SET_A, 'token_id_range': [TOKEN_LO, TOKEN_HI], 'ids': set_a},
                                 'B_english': {'texts': SET_B_TEXT, 'ids_truncated_to_64': set_b}},
                        'm1': 'mean over sequences and query positions 16..63 of attention[h, p, p-1]',
                        'profile': 'mean over sequences and query positions 47..63 of attention[h, p, p-d] for d in 0..47',
                        'permutation_test': {'n': 10000, 'seed': SEED_PERM, 'statistic': 'Spearman rho under random reassignment of predicted p1 to heads'}},
        'controls': ['set A removes lexical content as an explanation', 'set B checks the preference survives natural text',
                     'comparator heads preselected by rank, not searched', 'permutation null preregistered', 'bottom-3 reported as a negative prediction'],
        'not_claimed': ['causality', 'semantic meaning of any head', 'generalization beyond this model and these sequences'],
    }
    (HERE / 'PREDICTIONS.json').write_text(json.dumps(predictions, indent=1) + '\n')
    print('top3', top3, 'comparators', comparators, 'bottom3', bottom3)
    print('PREDICTIONS.json sha256', sha256_file(HERE / 'PREDICTIONS.json'))
    print('seconds', round(time.monotonic() - start, 1))


if __name__ == '__main__':
    main()
