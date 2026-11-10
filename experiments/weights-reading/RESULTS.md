# Result: the frozen weights-only ranking failed on this synthetic set

Run `run-20261107T084619Z`, 2026-11-07 08:46 UTC, M4 Pro, MPS, unchanged Qwen3.5-0.8B, eager attention, 20 sequences of 64 tokens, 12.2 seconds, peak RSS 2.37 GB. Predictions frozen in `PREDICTIONS.json` (sha256 `a5705fe4c1c84fd74ef2dd7e88a8778a79c20ca36aa50515a2661a7bde5e24d3`) and logged in the local experiment record (not distributed) before the probe. No prediction was changed after seeing results. All attempts are kept.

## Verdicts on the frozen predictions (pooled set A + B)

| Prediction | Measured | Criterion | Verdict |
| --- | --- | --- | --- |
| P1 rank: Spearman between frozen score and m1 over 48 heads | rho = −0.056, permutation p = 0.65 | supported if rho > 0.5 and p < 0.01; contradicted if rho < 0.2 | **contradicted** |
| P2: top-3 mean m1 vs median trio and bottom-3 | 0.068 vs 0.045 (1.51×) and 0.052 (1.31×) | supported if both ratios ≥ 2; contradicted if top ≤ median trio | inconclusive |
| P3: layer-19 mean m1 below median head and ≤ 0.5 × layer 3 | 0.0590 vs layer 3 0.0588, median head 0.051 | contradicted if layer 19 ≥ layer 3 | **contradicted** |

Per set: on random tokens (A) P1 rho = −0.13 (contradicted), P2 inconclusive, P3 contradicted (0.061 vs 0.050). On English (B) P1 rho = 0.07 (contradicted), P2 inconclusive, P3 inconclusive (0.057 vs 0.068). Each component alone: stand-in offset-1 contrast rho = 0.27, first-pass p1 rho = 0.24, high-frequency rotary energy share rho = −0.19. The high-frequency signal pointed the wrong way.

m1 is the mean attention probability a head places on the key one position back, over query positions 16 to 63.

## What the heads actually do (descriptive, not preregistered)

- The largest head-average m1 across these 20 sequences is 0.111 (L3.H3); set B's largest head average is 0.144. No strong average previous-token specialization was observed on this synthetic set. These averages do not exclude high attention at individual positions, conditional specialization or different behavior on other inputs.
- 41 of 48 heads nonetheless have offset 1 as their most-attended offset among 1 to 47, with small mass: a weak, nearly universal local tilt rather than a specialized head. Four heads peak at offset 47, two at 4, one at 23.
- Per layer, mean attention from query positions 47 to 63 goes 17 to 33 percent to the token itself, 3 to 5 percent to offset 1, 6 to 10 percent to offsets 2 to 4, 41 to 56 percent spread over offsets 5 to 47, and 9 to 20 percent beyond offset 47, which includes the first token.
- Measured top five by m1: L3.H3 0.111, L3.H0 0.094, L15.H1 0.076, L3.H6 0.072, L3.H1 0.066. Measured bottom: L23.H2 0.003, L23.H6 0.004, L3.H7 0.005. The frozen top-3 (L3.H1, L7.H0, L3.H0) contained two of the measured top five and one miss (L7.H0, about 0.045); the frozen bottom-3 and the layer-19 prediction were simply wrong.

## Interpretation

The frozen combined ranking failed its predeclared criterion on this set. Separately, the stand-in offset-1 contrast had a modest descriptive correlation of 0.27, while high-frequency rotary energy share had correlation -0.19. These secondary associations do not rescue the failed combined prediction or establish general predictive ability.

Across the six full-attention layers, no head had mean previous-token attention above 0.111 on this set. This measures average attention mass, not the proportion of times a head attends to the previous token, and does not prove an architectural absence. The first-pass description of weak positional signal was not a preregistered behavioral prediction and receives no success credit.

A post-hoc hypothesis, not tested: the 18 Gated DeltaNet layers carry kernel-4 causal depthwise convolutions and recurrent states that may provide some short-range mixing. Whether that explains the observed full-attention profiles is unknown. Reading those convolution weights would be a separate experiment, not an explanation established here.

## Corrections and caveats from independent review

- The reading computed weight-derived proxy quantities (projections of embedding rows through the layer's norms and projections). It did not compute activations conditioned on any text, but intermediate quantities were computed. The protocol's "no intermediate quantities" framing is withdrawn in favor of this description.
- Erratum to the frozen `PROTOCOL.md` (sha256 `ccb9d10625c2c661…`, left unchanged in the original local archive so its hash stands there; this public protocol copy has privacy edits and a different hash): the question says the validation sequences were "never shown to the reader". The reader wrote the set B passages and chose set A's seed. What was blind is the model's actual attention on them, not the sequences themselves.
- Uniform random token ids still select learned embeddings, so set A removes natural-language structure, not all lexical effects.
- The permutation p-value treats 48 heads as exchangeable although they share layers and key-value groups. It is a diagnostic null, not 48 independent replications. With rho near zero it did not affect any verdict.
- The reading-phase statement that layer-19 heads "cannot resolve offsets of one to three tokens" was the hypothesis that failed, not an architectural fact. Low energy in the highest rotary frequencies does not by itself prevent adjacent-offset discrimination: the remaining frequencies are nonzero, and content and earlier layers can supply the distinguishing information. The measurement shows layer 19 attends to the previous token as much as layer 3 does.
- Executed sources and frozen inputs (`validate.py`, `analysis.py`, `PREDICTIONS.json`, `PROTOCOL.md`) were copied beside the private run after execution, after confirming their hashes matched the run manifest; the copy is marked as post-execution in `archive-note.json`, and `artifact-sha256.json` inventories every file in the run directory. Private run files are owner-only. The checkpoint hash recorded by the probe matches the pinned base file; no model was modified and no further GPU work is planned for this experiment.
- The hash recorded before the freeze, `1b4b1dac…`, is `PREDICTIONS-v1-draft-not-frozen.json`, the first-pass draft that was replaced for the weights-only reasons recorded in `PROTOCOL.md` and the local experiment record (not distributed) before any probe ran.

## One concise finding

Contradicted for this frozen ranking: with the chosen weight-derived proxy, predicted and measured head rankings were essentially uncorrelated on 20 synthetic sequences, and the layer-19 prediction also failed. The top-versus-comparator result was inconclusive. This is a useful negative result for this method and set, not a disproof of weights reading or evidence that this model lacks previous-token specialization on other inputs. It does not evaluate the published editor's different models, tools or tasks.
