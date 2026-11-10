"""Period-agnostic repetition diagnostics over saved token streams (Claude, standard library only).

For each sample directory under a behavioral-replay run, reads tokens.jsonl (one JSON list of token ids per line,
as the TokenJournal wrote them), concatenates, and reports:
  - for every period p in 1..64, the longest run of consecutive exact repeats of a p-token block, and the best
    (period, repeats, first_end) over all p;
  - zlib compression ratio of the little-endian uint32 token-id stream (lower = more repetitive);
  - the most frequent 8-gram and its count.
Counts only; no text is decoded or written.

    python3 -I loop_diagnostics_stdlib.py <run dir> <report.json>
"""
import json
import struct
import sys
import zlib
from collections import Counter
from pathlib import Path


def longest_periodic_run(tokens, p):
    """Longest run of consecutive identical p-blocks at any alignment; returns (repeats, end_index)."""
    best, best_end = 0, None
    n = len(tokens)
    if n < 2 * p:
        return 0, None
    # tokens[i] == tokens[i-p] marks continuation of period p. A run of m such matches spans m+p tokens,
    # i.e. floor((m+p)/p) = 1 + m//p whole repeated blocks.
    run = 0
    for i in range(p, n):
        if tokens[i] == tokens[i - p]:
            run += 1
            repeats = 1 + run // p
            if repeats > best:
                best, best_end = repeats, i + 1
        else:
            run = 0
    return best, best_end


def diagnose(tokens):
    per_period = {}
    overall = (0, None, None)
    for p in range(1, 65):
        reps, end = longest_periodic_run(tokens, p)
        per_period[p] = reps
        if reps > overall[0]:
            overall = (reps, p, end)
    raw = struct.pack('<%dI' % len(tokens), *tokens) if tokens else b''
    ratio = len(zlib.compress(raw, 9)) / len(raw) if raw else None
    grams = Counter(tuple(tokens[i:i + 8]) for i in range(len(tokens) - 7))
    top = grams.most_common(1)[0] if grams else (None, 0)
    return {'tokens': len(tokens), 'max_repeats': overall[0], 'period_of_max': overall[1], 'end_of_max': overall[2],
            'repeats_at_period_16': per_period[16], 'any_period_le64_repeats_ge4': any(v >= 4 for v in per_period.values()),
            'zlib_ratio': ratio, 'top_8gram_count': top[1], 'distinct_8gram_fraction': len(grams) / max(1, len(tokens) - 7)}


def main(run_dir, report_path):
    run_dir = Path(run_dir)
    rows = []
    for sample in sorted(p for p in run_dir.iterdir() if p.is_dir() and p.name[:2].isdigit()):
        tokens = []
        for line in (sample / 'tokens.jsonl').read_text().splitlines():
            if line.strip():
                tokens.extend(json.loads(line))
        result = json.loads((sample / 'result.json').read_text()) if (sample / 'result.json').exists() else {}
        row = {'sample': sample.name, 'arm': result.get('arm'), 'id': result.get('id'), 'stop': result.get('stop'),
               'grading_status': result.get('grading_status'), **diagnose(tokens)}
        rows.append(row)
        print(json.dumps(row))
    Path(report_path).write_text(json.dumps({'run': run_dir.name, 'samples': rows}, indent=2) + '\n')


if __name__ == '__main__':
    main(*sys.argv[1:3])
