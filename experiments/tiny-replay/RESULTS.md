# Tiny replay result: integrity passed, behavior inconclusive

Completed 2026-11-07T07:38:06.210433+00:00 on the local M4 Pro. The pinned published p50c patch reconstructs exactly, and both baseline and edited models run locally. This establishes successful patch implementation and runtime portability only, not reproduction of the paper's benchmark improvement.

## Verified

- Full 1,746,942,600-byte model SHA-256 matches the pinned Hugging Face LFS object. The locally renamed shard is documented in README and the private summary.
- Patch archive and manifest hashes match the project's pinned upstream provenance inventory.
- Both BF16 base and edited tensor hashes match the upstream manifest. Changed entries: 65,260 in layer 15 and 64,928 in layer 19, totaling 130,188.
- A hash sweep of the entire loaded state found no other changed tensors. The base checkpoint on disk was never edited.
- Matched prompt token IDs, tokenizer, runtime, source precision, seed and decoding settings across each baseline/edit pair. Presence penalty precedes temperature/top-k/top-p in the inspected installed runtime.

## Measured smoke result

Four fixed synthetic prompts per variant, 256 generated tokens per prompt. All eight samples reached the token cap; none reached EOS, the 60-second time limit, or the defined exact repeated-block stop. Neither variant produced a nonempty final boxed answer before truncation. These are incomplete outputs, not evidence of zero mathematical ability or of model failure. A 256-token window cannot assess the original long-generation looping problem.

Baseline: 1024 tokens in 24.02 seconds (42.63 tokens/second). Edited: 1024 tokens in 21.89 seconds (46.79 tokens/second). This includes first-generation warmup and is not a rigorous speed benchmark. Peak process RSS was 4.11 GB; separate GPU allocation peak was not instrumented, so this is not total unified-memory peak. Disk footprint including isolated runtime and weights was 2.4 GiB.

## Preserved unsuccessful preparation

The initial CPU attempt produced three 47-token samples, each cut at approximately 60 seconds, and was stopped during the next sample rather than spending more time on incomplete outputs. Its results and manifest remain in `private/tiny-replay/cpu-partial/`. A BF16 MPS operation check passed, then the separately labeled MPS run completed all eight samples. No new packages were needed for the switch. The earlier initialization failures and their fixes are documented in README. No sample was silently discarded as successful.

## Scope and next decision

No email was read by a model, no external inference or paid API calls occurred, and no judging or production files were changed. This is an internal MLP patch replay, not output-head proof or independent discovery. The upstream evaluation imports an unreleased private package; full benchmark reproduction remains unimplemented. A longer, explicitly designed paired run could test behavior later, but none is started or promised by this result.

Detailed output tokens, prompts, hashes and versions are in the ignored local run artifacts described in README. The live adapter is `replay_mps.py`; `replay.py` preserves the CPU variant. Neither should be rerun over existing reports without first preserving them.

## Machine-readable grading

All eight completed MPS-run records are explicitly `truncated_ungraded` with `correct: null`, as are the three preserved partial CPU records. Raw parser booleans are separately retained as `exact_box_match`, and original files remain `results.raw.json`. This is a metadata correction after execution, not another run. Executed adapter snapshots and the original manifests preserve code provenance; see README and `grading-schema-update.json` for the revised-source hashes.

## Follow-up, 2026-11-07

A separately frozen, longer three-arm experiment has now completed. See `../behavioral-replay/RESULTS.md` for its narrow local result and controls; this does not change the inconclusive status of the original smoke run. The later audit found that this loader's generation configuration uses endoftext token 248044, whereas the tokenizer EOS and im_end are 248046. The policy was shared across variants but differs from conversational stopping; the original upstream stopping behavior has not been verified. Preserve this portability caveat when comparing results.
