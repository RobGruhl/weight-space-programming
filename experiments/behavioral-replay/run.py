"""Own offline MPS adapter. --prepare never constructs or loads a model."""
import os
os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', TOKENIZERS_PARALLELISM='false')
import argparse
import hashlib
import json
import platform
import resource
import shutil
import signal
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import transformers
from safetensors import safe_open
from transformers import AutoTokenizer, LogitsProcessor, Qwen3_5ForCausalLM, Qwen3_5TextConfig, StoppingCriteria
from metrics import ARMS, PROMPTS, SUFFIX, grade, repetition

ROOT = Path(__file__).resolve().parents[2]
SOURCE = Path(__file__).resolve().parent
BASE = ROOT / 'private/tiny-replay/model'
PATCH = ROOT / 'references/upstream/exp8_math/runs/p50c/edit'
DEST = ROOT / 'private/behavioral-replay'
MODEL_SHA = '04b1c301231dd422b8860db31311ab2721511346a32cb1e079c4c4e5f1fe4696'
SEED = 20261107
CAP = 8192

def sha(data):
    return hashlib.sha256(data).hexdigest()

def file_sha(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()

def tensor_sha(t):
    return sha(t.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())

def write(path, obj):
    path.write_text(json.dumps(obj, indent=2) + '\n')

def resources():
    # ru_maxrss is bytes on this macOS runtime and a lifetime peak.
    return {'rss_lifetime_peak_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            'mps_current_allocated_bytes': torch.mps.current_allocated_memory(),
            'mps_driver_allocated_bytes': torch.mps.driver_allocated_memory()}

def construct_control(base, edited, columns, seed):
    rng = np.random.default_rng(seed)
    permutation = rng.permutation(base.shape[0])
    delta = edited.float() - base.float()
    shuffled = base.float().clone()
    shuffled[:, columns] += delta[permutation][:, columns]
    shuffled = shuffled.to(base.dtype)
    realized = shuffled.float() - base.float()
    target_norm = torch.linalg.vector_norm(delta).item()
    control_norm = torch.linalg.vector_norm(realized).item()
    outside = torch.ones(base.shape[1], dtype=torch.bool)
    outside[columns] = False
    assert torch.equal(shuffled[:, outside], base[:, outside])
    assert abs(control_norm / target_norm - 1) <= .01
    report = {'seed': seed, 'permutation_sha256': sha(permutation.astype('<i8').tobytes()),
              'p50c_delta_frobenius': target_norm, 'control_delta_frobenius': control_norm,
              'relative_norm_mismatch': control_norm / target_norm - 1,
              'delta_cosine': torch.dot(delta.flatten(), realized.flatten()).item() / (target_norm * control_norm),
              'selected_columns': columns.tolist(), 'selected_entries': base.shape[0] * len(columns),
              'p50c_changed_entries': int(torch.count_nonzero(delta)),
              'control_changed_entries': int(torch.count_nonzero(realized)),
              'control_changed_columns': int(torch.count_nonzero(realized.ne(0).any(dim=0))),
              'control_changed_rows': int(torch.count_nonzero(realized.ne(0).any(dim=1))),
              'control_sha256': tensor_sha(shuffled)}
    return shuffled, report

def prepare_patches():
    manifest = json.loads((PATCH / 'manifest.json').read_text())
    arrays = np.load(PATCH / 'edit.npz', allow_pickle=False)
    weights, report = {}, []
    with safe_open(BASE / 'model.safetensors', framework='pt', device='cpu') as f:
        for i, item in enumerate(manifest['tensors']):
            base = f.get_tensor(item['name'])
            assert base.dtype == torch.bfloat16
            assert tensor_sha(base) == item['base_sha256']
            edited_bits = base.view(torch.uint16).numpy().copy()
            edited_bits.reshape(-1)[arrays[f't{i}_idx']] = arrays[f't{i}_val']
            edited = torch.from_numpy(edited_bits).view(torch.bfloat16)
            assert tensor_sha(edited) == item['edited_sha256']
            assert int(torch.count_nonzero(base != edited)) == item['changed_entries']
            columns = torch.nonzero(base.ne(edited).any(dim=0)).flatten()
            assert len(columns) == 64
            control, detail = construct_control(base, edited, columns, SEED + i)
            name = item['name'].replace('model.language_model.', 'model.', 1)
            weights[name] = {'baseline': base, 'p50c': edited, 'shuffled': control}
            report.append(dict(name=name, base_sha256=tensor_sha(base), p50c_sha256=tensor_sha(edited), **detail))
    return weights, report

class Presence(LogitsProcessor):
    def __init__(self, n):
        self.n = n

    def __call__(self, ids, scores):
        for row in range(ids.shape[0]):
            scores[row, torch.unique(ids[row, self.n:])] -= 1.5
        return scores

class Bound(StoppingCriteria):
    def __init__(self, n, deadline):
        self.n, self.deadline, self.start = n, deadline, time.monotonic()
        self.reason, self.telemetry = None, []

    def __call__(self, ids, scores, **kwargs):
        now = time.monotonic()
        count = ids.shape[1] - self.n
        if count % 32 == 0 or count == 1:
            sample = dict(tokens=count, seconds=now-self.start, **resources())
            self.telemetry.append(sample)
            if sample['rss_lifetime_peak_bytes'] > 12 * 1024**3 or sample['mps_driver_allocated_bytes'] > 14 * 1024**3:
                self.reason = 'memory_safety_limit'
        if now - self.start >= 210:
            self.reason = 'time_limit'
        if now >= self.deadline:
            self.reason = 'whole_run_time_limit'
        return self.reason is not None

class TokenJournal:
    """Prompt is recorded separately; flush generated tokens every 32 tokens."""
    def __init__(self, path):
        self.file = path.open('x')
        self.prompt = True
        self.tokens = []

    def put(self, value):
        if self.prompt:
            self.prompt = False
            return
        tokens = value.reshape(-1).tolist()
        self.tokens.extend(tokens)
        self.file.write(json.dumps(tokens) + '\n')
        if len(self.tokens) % 32 == 0:
            self.file.flush()

    def end(self):
        self.file.flush()

def watchdog(signum, frame):
    raise TimeoutError('2700-second outer wall-clock watchdog')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--gpu-released', action='store_true', help='Only after explicit operator resource approval')
    args = parser.parse_args()
    if not args.prepare and not args.gpu_released:
        parser.error('Generation requires explicit GPU release; use --prepare for CPU-only audit')
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')
    out = DEST / (('preparation-' if args.prepare else 'run-') + stamp)
    out.mkdir(parents=True, exist_ok=False)
    for name in ['run.py', 'metrics.py', 'PROTOCOL.md']:
        shutil.copyfile(SOURCE / name, out / name)
    manifest = {'status': 'started', 'start_utc': stamp, 'prepare_only': args.prepare,
                'model_revision': '2fc06364715b967f1860aea9cf38778875588b17',
                'source_hashes': {name: file_sha(SOURCE / name) for name in ['run.py', 'metrics.py', 'PROTOCOL.md']},
                'patch_hashes': {name: file_sha(PATCH / name) for name in ['edit.npz', 'manifest.json']},
                'python': platform.python_version(), 'torch': torch.__version__, 'transformers': transformers.__version__,
                'seed': 0, 'control_seed_base': SEED, 'max_new_tokens': CAP, 'seconds_per_sample': 210,
                'whole_run_seconds': 2700, 'device': 'mps', 'temperature': .6, 'top_p': .95, 'top_k': 20,
                'presence_penalty': 1.5, 'thinking': True, 'prompts': PROMPTS, 'suffix': SUFFIX,
                'order': [[ARMS[(j+i) % 3] for j in range(3)] for i in range(4)],
                'dtype': 'original BF16 plus FP32 tensors; no quantization', 'cost': 'local only, no paid API'}
    write(out / 'manifest.json', manifest)
    print(str(out), flush=True)
    results = []
    try:
        manifest['model_files'] = {p.name: file_sha(p) for p in BASE.iterdir() if p.is_file()}
        assert manifest['model_files']['model.safetensors'] == MODEL_SHA
        weights, integrity = prepare_patches()
        write(out / 'patch-integrity.json', integrity)
        if args.prepare:
            manifest['status'] = 'prepared_no_model_loaded'
            return
        assert torch.backends.mps.is_available()
        deadline = time.monotonic() + 2700
        signal.signal(signal.SIGALRM, watchdog)
        signal.alarm(2700)
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
        model.eval().to('mps')
        before = {k: tensor_sha(v) for k, v in model.state_dict().items()}
        tokenizer = AutoTokenizer.from_pretrained(BASE, local_files_only=True, trust_remote_code=False)
        eos_ids = model.generation_config.eos_token_id
        eos_ids = eos_ids if isinstance(eos_ids, list) else [eos_ids]
        manifest['eos_token_ids'] = eos_ids
        write(out / 'manifest.json', manifest)
        for i, (ident, question, answer) in enumerate(PROMPTS):
            for arm in manifest['order'][i]:
                if time.monotonic() >= deadline:
                    raise TimeoutError('Whole-run budget exhausted before next sample')
                sample_dir = out / f'{len(results):02d}-{ident}-{arm}'
                sample_dir.mkdir()
                with torch.no_grad():
                    for name, choices in weights.items():
                        model.state_dict()[name].copy_(choices[arm])
                after = {k: tensor_sha(v) for k, v in model.state_dict().items()}
                for name, value in after.items():
                    assert value == (tensor_sha(weights[name][arm]) if name in weights else before[name]), name
                write(sample_dir / 'state-integrity.json', {'verified': True, 'changed_tensors': [k for k in before if before[k] != after[k]], 'state_hashes': after})
                torch.manual_seed(0)
                prompt = tokenizer.apply_chat_template([{'role': 'user', 'content': question + SUFFIX}], tokenize=False, add_generation_prompt=True, enable_thinking=True)
                inputs = tokenizer(prompt, return_tensors='pt').to('mps')
                n = inputs.input_ids.shape[1]
                write(sample_dir / 'input.json', {'id': ident, 'arm': arm, 'expected': answer, 'rendered_prompt': prompt, 'prompt_sha256': sha(prompt.encode()), 'input_ids': inputs.input_ids[0].tolist()})
                stop, journal = Bound(n, deadline), TokenJournal(sample_dir / 'tokens.jsonl')
                start = time.monotonic()
                try:
                    with torch.inference_mode():
                        ids = model.generate(**inputs, max_new_tokens=CAP, do_sample=True, temperature=.6, top_p=.95, top_k=20, logits_processor=[Presence(n)], stopping_criteria=[stop], pad_token_id=tokenizer.eos_token_id, use_cache=True, streamer=journal)
                    torch.mps.synchronize()
                    tokens = ids[0, n:].tolist()
                    assert tokens == journal.tokens
                    eos = bool(tokens and tokens[-1] in eos_ids)
                    text = tokenizer.decode(tokens, skip_special_tokens=False)
                    reason = 'eos' if eos else (stop.reason or ('token_limit' if len(tokens) == CAP else 'unknown_stop'))
                    row = {'arm': arm, 'id': ident, 'expected': answer, 'tokens': len(tokens), 'output_ids': tokens, 'output': text, 'eos': eos, 'stop': reason, 'safety_criterion': stop.reason, 'seconds': time.monotonic()-start, **grade(text, answer, eos), **repetition(tokens), **resources()}
                    write(sample_dir / 'result.json', row)
                    results.append(row)
                    write(out / 'results.json', results)
                    print(json.dumps({k:v for k,v in row.items() if k not in ['output', 'output_ids']}), flush=True)
                    if stop.reason in ['memory_safety_limit', 'whole_run_time_limit']:
                        raise RuntimeError('Aborting remaining samples after ' + stop.reason)
                except BaseException:
                    write(sample_dir / 'error.json', {'error': traceback.format_exc(), 'partial_tokens': journal.tokens, 'seconds': time.monotonic()-start})
                    raise
                finally:
                    journal.file.close()
                    write(sample_dir / 'resource-samples.json', stop.telemetry)
                del ids, inputs
                torch.mps.empty_cache()
        manifest['status'] = 'complete'
    except BaseException:
        manifest['status'] = 'failed_or_interrupted'
        write(out / 'error.json', {'error': traceback.format_exc()})
        raise
    finally:
        signal.alarm(0)
        manifest['finished_utc'] = datetime.now(timezone.utc).isoformat()
        manifest['completed_samples'] = len(results)
        write(out / 'manifest.json', manifest)

if __name__ == '__main__':
    main()
