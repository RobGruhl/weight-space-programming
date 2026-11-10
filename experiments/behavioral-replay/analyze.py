"""Post-hoc diagnostics; never changes executed grading, raw tokens or generation."""
import argparse
import hashlib
import json
import struct
import zlib
from pathlib import Path

def diagnostics(tokens, text, eos):
    maxima = {}
    for period in range(1, 65):
        maximum = 0
        for offset in range(period):
            previous, run = None, 0
            for end in range(offset + period, len(tokens) + 1, period):
                block = tokens[end-period:end]
                run = run + 1 if block == previous else 1
                previous = block
                maximum = max(maximum, run)
        maxima[str(period)] = maximum
    encoded = struct.pack('<' + 'I' * len(tokens), *tokens)
    closed = '</think>' in text
    import re
    boxes = re.findall(r'\\boxed\{([^{}]*)\}', text)
    has_box = bool(boxes and boxes[-1].strip())
    if not eos:
        category = 'censored'
    elif has_box and not closed:
        category = 'eos_box_present_without_closing_think'
    elif not has_box:
        category = 'eos_no_nonempty_box'
    else:
        category = 'eos_box_and_closing_think_present'
    return {'maximum_exact_adjacent_repeats_by_period': maxima,
            'periods_with_four_adjacent_repeats': [int(p) for p, n in maxima.items() if n >= 4],
            'compression': {'encoding': 'little-endian uint32 token IDs', 'algorithm': 'zlib level 9',
                            'raw_bytes': len(encoded), 'compressed_bytes': len(zlib.compress(encoded, 9)),
                            'compressed_over_raw': len(zlib.compress(encoded, 9)) / len(encoded) if encoded else None},
            'eos_format_category': category}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('run', type=Path)
    args = parser.parse_args()
    run = args.run
    manifest = json.loads((run / 'manifest.json').read_text())
    if manifest['status'] != 'complete':
        parser.error('Only analyze a completed run; preserve unfinished attempts separately')
    results = json.loads((run / 'results.json').read_text())
    from transformers import AutoTokenizer
    root = Path(__file__).resolve().parents[2]
    tokenizer = AutoTokenizer.from_pretrained(root / 'private/tiny-replay/model', local_files_only=True, trust_remote_code=False)
    eos = tokenizer.eos_token_id
    im_end = tokenizer.convert_tokens_to_ids('<|im_end|>')
    analysis = {'classification': 'post-hoc exploratory diagnostics requested during generation; no generation change',
                'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                'raw_results_sha256': hashlib.sha256((run / 'results.json').read_bytes()).hexdigest(),
                'zlib_version': zlib.ZLIB_VERSION,
                'tokenizer_eos_id': eos, 'im_end_id': im_end, 'equal': eos == im_end,
                'generation_eos_ids': manifest['eos_token_ids'],
                'records': [{'arm': r['arm'], 'id': r['id'], **diagnostics(r['output_ids'], r['output'], r['eos'])} for r in results]}
    output = run / 'exploratory-analysis.json'
    with output.open('x') as f:
        json.dump(analysis, f, indent=2)
        f.write('\n')
    # Keep the actual analysis implementation beside its output, separately from frozen run code.
    with (run / 'analyze.executed.py').open('x') as f:
        f.write(Path(__file__).read_text())
    print(output)

if __name__ == '__main__':
    main()
