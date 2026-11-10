"""Separate post-hoc termination and file-integrity audit; no model load."""
import argparse
import hashlib
import json
from pathlib import Path

def termination(tokens, im_end, generation_eos):
    first = tokens.index(im_end) if im_end in tokens else None
    return {'first_im_end_index_zero_based': first,
            'tokens_through_first_im_end': first + 1 if first is not None else None,
            'tokens_after_first_im_end': len(tokens) - first - 1 if first is not None else None,
            'suffix_after_first_im_end': tokens[first+1:] if first is not None else [],
            'total_tokens': len(tokens), 'last_token_is_generation_eos': bool(tokens and tokens[-1] in generation_eos)}

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('run', type=Path)
    run = parser.parse_args().run
    exploratory = json.loads((run / 'exploratory-analysis.json').read_text())
    results = json.loads((run / 'results.json').read_text())
    report = {'classification': 'post-hoc termination audit; original results unchanged',
              'source_sha256': digest(Path(__file__)), 'upstream_stop_policy_verified': False,
              'im_end_id': exploratory['im_end_id'], 'generation_eos_ids': exploratory['generation_eos_ids'],
              'records': [{'id': r['id'], 'arm': r['arm'], **termination(r['output_ids'], exploratory['im_end_id'], exploratory['generation_eos_ids'])} for r in results]}
    with (run / 'termination-audit.json').open('x') as f:
        json.dump(report, f, indent=2)
        f.write('\n')
    with (run / 'audit_termination.executed.py').open('x') as f:
        f.write(Path(__file__).read_text())
    inventory = {str(p.relative_to(run)): digest(p) for p in sorted(run.rglob('*')) if p.is_file()}
    with (run / 'artifact-sha256.json').open('x') as f:
        json.dump(inventory, f, indent=2)
        f.write('\n')

if __name__ == '__main__':
    main()
