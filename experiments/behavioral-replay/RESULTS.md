# One bounded completion rescue; not a benchmark result

The exact published p50c patch completed all four development questions correctly. Baseline completed three correctly and exhausted 8,192 tokens on the rectangle-area question while repeating a phrase; p50c completed that question correctly in 1,678 tokens. The norm-matched shuffled-direction control also solved the rectangle question, but exhausted its budget on addition. This is one baseline-relative rescue in four previously used synthetic questions, with one seed. It does not establish general mathematical improvement, a unique repetition mechanism, independent edit discovery, or full benchmark reproduction.

**Termination caveat:** local generation stopped at `<|endoftext|>`, not the tokenizer's `<|im_end|>` EOS. Every completed sample emitted exactly two tokens after `<|im_end|>`; completion categories are unaffected here. Actual upstream stopping behavior remains unverified. Details and per-output positions are below.

Run: `20261107T075414.954099Z`, completed 2026-11-07 08:08:21 UTC. Local M4 Pro, original mixed BF16/FP32 checkpoint, Transformers/MPS. Twelve of twelve preregistered samples completed execution; no generation error, safety stop, sample time cutoff, or outer watchdog cutoff occurred. Total wall time was 14 minutes 7 seconds, including setup. No private email, external inference, paid API, training, or production changes were involved. GPU resources were released after the process exited normally.

## Paired outcomes

Numbers below are generated tokens. “Correct” means EOS plus a correct final boxed integer after the closing thinking marker, not a judgment that every sentence of the explanation is sound. “Censored” is ungraded, not an incorrect capability label.

| Development question | Baseline | Exact p50c | Shuffled delta |
| --- | ---: | ---: | ---: |
| 17 + 28 | Correct, 2,275 | Correct, 635 | Censored, 8,192 |
| 3x + 7 = 22 | Correct, 1,053 | Correct, 1,393 | Correct, 3,323 |
| Rectangle area, 8 by 5 | Censored, 8,192 | Correct, 1,678 | Correct, 5,037 |
| 2 + 2 | Correct, 1,794 | Correct, 2,643 | Correct, 1,154 |

- Baseline: 3 correct completed answers, 0 wrong completed answers, 0 completed missing answers, 1 token-censored output.
- p50c: 4 correct completed answers, 0 wrong completed answers, 0 completed missing answers, 0 censored outputs.
- Shuffled: 3 correct completed answers, 0 wrong completed answers, 0 completed missing answers, 1 token-censored output.
- All ten EOS outputs closed thinking and supplied the expected final box; no EOS-with-box-but-no-closing-think cases occurred. Both censored outputs lacked a closing thinking marker and a nonempty final answer. Raw `exact_box_match`, semantic correctness and answer-format compliance remain separate fields.
- p50c was longer than baseline on algebra and 2 + 2. It was not uniformly faster or shorter. Aggregate tokens/time were baseline 13,314 / 296.04 s, p50c 6,349 / 132.32 s, shuffled 17,706 / 396.13 s. This is a censored small-sample workload comparison, not a runtime speed benchmark.

The first 256 output tokens of all eight baseline/p50c samples exactly match their earlier smoke counterparts. The original smoke's failure to finish was therefore a short observation window, not evidence that these tasks were all beyond the model.

## Repetition: preregistered versus exploratory

The original exact-16-token-block detector flagged none of the twelve outputs. It misses periods that do not divide 16. At independent reviewer's suggestion during the run, a separate post-hoc analysis checked every period from 1 through 64 and zlib compression of little-endian uint32 token IDs, for every arm equally. Generation, stopping, executed code and raw grading were unchanged. These are exploratory diagnostics, not a retrospectively substituted primary endpoint.

| Output | Longest exact adjacent run | Compression ratio |
| --- | --- | ---: |
| Baseline rectangle, censored | 253 copies of a 12-token block | 0.164 |
| Shuffled addition, censored | 474 copies of a 10-token block | 0.086 |
| Shuffled algebra, correct completion | 54 copies of a 14-token block | 0.150 |

No other sample had four consecutive identical blocks of any tested period; none of the p50c samples did. For example, the censored baseline rectangle output repeatedly says “I'll make sure to check the prompt again.” This establishes observed exact repetition in that local trace, beyond merely equating truncation with looping. Shuffled algebra also demonstrates that a repetitive run can recover and answer correctly. Compression ratios depend on token representation and length and are not calibrated looping probabilities.

## Integrity and control interpretation

Model and p50c hashes match the pinned upstream manifest and earlier independent audits. Every arm installation checked hashes for the entire loaded state; only the intended two tensors changed, and original weights were restored before each new arm. All rendered inputs and seeds were matched. At the original local audit, frozen run.py, metrics.py and PROTOCOL.md hashes matched the executed snapshots. This public export edits coordination wording in run.py and PROTOCOL.md, so their public bytes differ; see ../../PUBLICATION.md.

The shuffled control used independently frozen seeds 20261107 and 20261108 (tensor manifest index, not layer number). Actual BF16 delta norm differences relative to p50c were +0.00855% and +0.00159%; directional cosines were +0.00798 and −0.02026. Both controls changed only the same 64 columns per tensor. Actual changed-entry counts were 65,099 and 64,718, compared with 65,260 and 64,928 for p50c. There was no outcome-driven norm adjustment, seed selection or rerun. The separate geometry worker's seed-plus-layer control is a different artifact and was not run here.

The published alpha=6 edit reverses the selected direction's existing component to approximately −5 times its original value. It is not a projection to zero. Row shuffling retains most of that original component while adding a near-orthogonal perturbation of similar magnitude. Therefore shuffled harm may reflect generic off-direction damage; it does not uniquely prove that the published direction encodes repetition. The primary evidence here is p50c versus baseline on these exact traces, not the claim that a bad control establishes mechanism.

## Termination and resource audit

The tokenizer's EOS ID and `<|im_end|>` ID are both 248046. However, the loaded model generation configuration's stopping EOS is 248044 (`<|endoftext|>`). These are not equal. Every completed sample emitted `<|im_end|>`, then exactly two more tokens (newline and `<|endoftext|>`) before stopping. Neither censored sample contained `<|im_end|>`. All arms and the previous smoke used the same generation EOS policy. Consequently stopping at `<|im_end|>` instead would have removed exactly two tokens from each completed trace and changed none of these completion categories, but this runtime policy must not be presented as verified upstream vLLM termination parity.

| Completed output | First im_end, zero-based index | Tokens through im_end | Total tokens |
| --- | ---: | ---: | ---: |
| Addition baseline | 2,272 | 2,273 | 2,275 |
| Addition p50c | 632 | 633 | 635 |
| Algebra baseline | 1,050 | 1,051 | 1,053 |
| Algebra p50c | 1,390 | 1,391 | 1,393 |
| Algebra shuffled | 3,320 | 3,321 | 3,323 |
| Geometry p50c | 1,675 | 1,676 | 1,678 |
| Geometry shuffled | 5,034 | 5,035 | 5,037 |
| 2 + 2 baseline | 1,791 | 1,792 | 1,794 |
| 2 + 2 p50c | 2,640 | 2,641 | 2,643 |
| 2 + 2 shuffled | 1,151 | 1,152 | 1,154 |

Reported elapsed times include the two extra tokens; per-token timing was not captured, so their latency cannot be subtracted exactly. `termination-audit.json` preserves the indices and suffix token IDs separately from raw results.

Maximum sampled MPS tensor allocation was 2,147,092,224 bytes; maximum sampled driver allocation was 3,508,305,920 bytes. Process lifetime peak RSS was 2,427,043,840 bytes. These overlap on unified memory and must not be added. Sampling every 32 generated tokens is not a guaranteed instantaneous GPU peak. No sample reached the 210-second cap (the longest was 190.17 seconds). No additional installs or downloads were needed.

## Reproducibility and limits

Ignored run artifacts contain the manifest, source snapshots/hashes, checkpoint/patch hashes, full state checks, rendered inputs, every generated token, raw output, resource samples and separately hashed exploratory analysis. The preparation-only attempt is retained separately; it did not load a model. One later test invocation failed under the read-only sandbox because PyTorch could not create its temporary cache; the same eight tests passed under authorized local execution with writable runtime temporary directories. This was not a generation failure. The initial five tests also passed before inference. No failed generation or unfavorable sample was discarded.

The final suite includes ten tests, including termination-position tests. The complete private artifact inventory is `artifact-sha256.json`; model bytes are covered by the pinned full-model hash in the manifest. All files in this experiment's private directory are owner-readable/writable only (0600), directories owner-only (0700), and Git exclusion was verified. No further GPU generation is planned.

Four development questions, one generation seed, one shuffled seed per layer, an 8,192-token cap and a different runtime cannot estimate benchmark accuracy or reliable effect size. No fresh holdout was consumed, and no new edit was tested. A new confirmation experiment would require a separately frozen prompt set and design; none is implied or scheduled by this report.
