# Weights-reading experiment: positional heads of Qwen3.5-0.8B from weights alone

Frozen 2026-11-07 before any validation forward pass. Reader: the assistant in this session (Claude Fable 5.1). This is not the published editor model, and no identity continuity with it is claimed. Local compute only. This public protocol copy removes private coordination metadata; its bytes differ from the original frozen protocol hash.

## Question

Given only the weights and architecture of Qwen3.5-0.8B, can the reader rank its 48 full-attention heads by how much they attend to the immediately preceding token, and identify a layer whose heads cannot do so, such that the model's actual attention patterns on sequences never shown to the reader confirm the ranking?

## What the reader receives

- Weights: `embed_tokens.weight` (rows used as stand-in input vectors and to form an input second-moment matrix), and for each full-attention layer (3, 7, 11, 15, 19, 23) `input_layernorm.weight`, `self_attn.q_proj.weight`, `self_attn.k_proj.weight`, `self_attn.q_norm.weight`, `self_attn.k_norm.weight`. Tensor hashes are in `predicted-profiles-v1-flat.json`.
- Architecture: `config.json` (8 query heads, 2 key heads shared four to one, head dim 256, rotary on the first 64 dims with theta 1e7 and 32 frequencies, RMSNorm with `1 + weight` gains, query/gate split in `q_proj`) and the installed Transformers modeling code, reimplemented in `wsr.py` and verified against a randomly initialized library attention module in `test_reading.py`.
- Tokenizer: used only to build validation set B from fixed English passages written by the reader.
- Not received: activations from any text, model outputs, labels, probes, behavioral examples.

## Disclosure of prior knowledge

The reader knows from public research that transformers commonly contain previous-token heads and that rotary embeddings make the query-key score depend on relative position through fixed frequencies, so only high frequencies can distinguish offsets of one, two and three. That knowledge motivates the observables. Which heads in this model carry high-frequency rotary energy, and how the heads rank, is not known to the reader and is the falsifiable content. The published p50c repetition claim is not used. The reader has read the upstream blog and logs; none of them discuss this model's attention heads.

## Reading, as it actually happened

1. First pass (`read_weights.py`, output kept as `predicted-profiles-v1-flat.json`): for unrelated stand-in query and key vectors (384 embedding rows for token ids drawn from 1000 to 60000 with seed 20261107, passed through the input norm, the projections and the per-head norms), the closed-form rotary score at offsets 0 to 47, softmax over offsets, mean mass per offset. Result: mass at offset 1 between 0.019 and 0.027 for every head, where uniform is 0.021. The embedding-row proxy carries almost no content-independent positional signal. A ranking on these differences alone would be mostly noise, so the draft prediction file from this pass was not frozen; it is kept as `PREDICTIONS-v1-draft-not-frozen.json` for the record.
2. Second pass (`diagnose_weights.py`, output `weights-diagnostics.json`): per head, query and key energy per dimension after the norm gains under two input assumptions (isotropic, and the uncentered second moment of 4096 normalized embedding rows); the share of query-key energy in rotary dims; the share in the four highest rotary frequencies (`hf4`); and from the stand-in pairs the mean-score contrast at offset 1 (`contrast1`) and the fraction of pairs whose best offset is 1. Findings: no head exceeds the uniform 25% rotary share; all eight layer-19 heads put under 1.5% of rotary energy into the four highest frequencies against a uniform 12.5%; layer 3 and 7 heads keep the most; L3.H1 and L3.H2 have the largest contrast1. The pair-level fraction is dominated by an endpoint artifact of slowly varying low frequencies and is discarded.
3. Frozen score (`freeze_predictions.py`): mean of the two ranks (contrast1 and hf4), tie-break on contrast1. Both signals are weights-only. The combination was fixed before any validation output existed. Component correlations are reported separately as secondary.

## Predictions (exact heads in `PREDICTIONS.json`, hashed in the local experiment record (not distributed) before probing)

- P1, ranking: Spearman correlation between the frozen score and measured m1 across all 48 heads, pooled set. Supported if rho > 0.5 with one-sided permutation p < 0.01 (10,000 permutations, seed 20261107). Contradicted if rho < 0.2. Otherwise inconclusive.
- P2, top versus comparators: the three top-ranked heads have mean measured m1 at least twice that of the three median-ranked heads (ranks 24 to 26 of 48) and at least twice that of the bottom three. Contradicted if the top three do not exceed the median trio at all. Otherwise inconclusive.
- P3, layer 19 is non-local: mean m1 over the eight layer-19 heads is below the median head m1 and at most half the mean m1 of the eight layer-3 heads. Contradicted if layer 19's mean m1 is at least layer 3's. Otherwise inconclusive.

The reading does not predict the absolute level of m1; the stand-in proxy understates any mechanism that depends on the residual-stream mean built by earlier layers. The maximum measured m1 is reported descriptively.

## Validation

Unchanged model, eager attention, MPS, no sampling. Twenty sequences of 64 tokens: set A is ten random-token sequences (ids from the same range, seed 20261108), set B is ten fixed English passages truncated to 64 tokens. Both are fixed in `PREDICTIONS.json`. Attention probabilities are captured by forward hooks on the six attention modules. m1 for a head is the mean over sequences and query positions 16 to 63 of the probability on the key one position back. The mean profile uses query positions 47 to 63 so every offset 0 to 47 exists. `validate.py` refuses to run unless the hash of `PREDICTIONS.json` passed on its command line matches the file, and records model and prediction hashes in its manifest. Secondary reports: each component alone, set A and set B separately, argmax offsets, per-layer means.

## Controls

Set A removes lexical content as an explanation; set B checks survival on natural text. Comparator heads are preselected by rank, not searched. The permutation null is preregistered. Bottom-3 and layer-19 are negative predictions.

## What is not claimed

No causal claim; attention probabilities are a correlational readout. No claim about what any head means. No claim beyond this model and these sequences. If predictions fail, the failure is reported and the prediction file is not revised.

## Resources

Reading: CPU, two threads, seconds, under 2 GB. Validation: one forward pass per sequence on MPS, about 2 GB, a few minutes. Raw per-sequence statistics go to ignored `private/weights-reading/`; aggregates to this directory.
