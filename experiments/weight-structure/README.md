# Published p50c weight geometry

Analysis and intervention freeze: 2026-11-07T07:53:03Z. No local behavioral outcomes were inspected by this worker. The source final report and experiment history were inspected, including author-reported behavioral results, so the proposals are source-informed, not outcome-blind to the original publication. No claim of a new behavioral discovery is made.

## Provenance and method

Source checkout: `4d09ba6d6a7e891dd8063974dfe81b3d78e3494f`. Public patch: `references/upstream/exp8_math/runs/p50c/edit/edit.npz`, SHA-256 `e92a92aafd1f1068b4fd94aa0fbe081ba3ddea2047ee7bcac7161d861be299e7`. Local base model: Qwen3.5-0.8B revision `2fc06364715b967f1860aea9cf38778875588b17`, file SHA-256 `04b1c301231dd422b8860db31311ab2721511346a32cb1e079c4c4e5f1fe4696` supplied by the independently verified parent audit; this analysis did not rehash the full model. Only the two relevant tensors are loaded through safetensors. NPZ uses allow_pickle=False. No upstream scripts are executed.

Reproduce from repository root:

    private/tiny-replay/venv/bin/python experiments/weight-structure/analyze.py

CPU only, at most two numeric-library threads, no inference or model download. Aggregate results are in geometry.json. Float64 SVD is applied to the 1024 x 64 BF16-realized delta per layer. The leading left singular vector is estimated independently from the patch, with its largest absolute coordinate positive. Its sign has no semantic interpretation. Raw directions, deltas, permutations, and intervention arrays are saved only in ignored private/weight-structure/geometry.npz. Initial raw archive SHA-256: `667891c6db76f16d781434e11452a2493afaaa5a12016cc1f7f3d219c8932acf`.

## Independently measured geometry

- Layer 15: whole-matrix relative Frobenius change 10.63618%; selected-column relative change 84.97793%. Delta norm 1.748888. Leading singular component explains 99.999454% of delta energy.
- Layer 19: whole-matrix relative Frobenius change 8.10628%; selected-column relative change 62.23825%. Delta norm 1.398763. Leading singular component explains 99.999138% of delta energy.
- Both deltas have 64 numerical singular values at float64 tolerance, but one component explains more than 99.99% of energy. Formal rank 64 is compatible with BF16 rounding of an approximately rank-one edit, not evidence for 64 substantive mechanisms.
- Leading output directions have absolute cosine 0.849005 across layers. Only 3 selected input-column indices overlap. Coordinate alignment across layers is mathematically well-defined here but does not establish equal semantic meaning.
- Per-column delta norms range 0.046416–0.529634 in layer 15 and 0.022435–0.640488 in layer 19. Full column-wise norms and before/after projections are in geometry.json.
- Fitting delta = −alpha outer(u, uᵀW_selected) gives alpha 6.000321 and 6.000173. Fixed alpha 6 relative residuals are 0.2348% and 0.2944%, consistent with quantized approximately rank-one arithmetic. The fit uses a direction estimated from the delta itself, not independent hidden-state evidence.

The source final report states W'=W−6 outer(u, mask64*(uᵀW)), calling u a repetition direction estimated from matched contexts. The actual artifact agrees with this arithmetic. It is **not literal orthogonal projection removal**: selected-column projections become approximately −5.00032 and −5.00017 times their originals. Orthogonal components are approximately preserved, while the chosen component is reversed and amplified. Alpha 1 would remove it; alpha 6 overshoots. Untouched columns preserve their baseline projections. No claim about net activation-conditional MLP output follows from weights alone.

The semantic claim that u encodes repetition remains author-reported. We have not independently reconstructed the activation contrast, proven neuron semantics, or established causal behavioral specificity. Small discrepancies from source-manifest relative norms arise from exact BF16-realized differences and float64 computation; these are not alternate patches.

## Frozen negative control

Use the realized published delta D=BF16(W_published)−W_base. Independently for layer L, generate permutation P with np.random.default_rng(20261107+L).permutation(1024). Apply the same row permutation to all 64 selected columns: W_control=BF16(W_base+P D). All other weights remain baseline. No seed selection or retry based on behavioral results.

This preserves selected column support, singular values and each column update norm before final BF16 rounding. It disrupts the residual output direction while preserving the column coefficient pattern. Final norm ratios are 0.999909/0.999947; maximum per-column norm drift is 0.104%/0.078%. Leading-direction absolute cosines with the original are 0.02955/0.00819. Changed-entry counts differ slightly due to rounding. This is not an exact causal isolator: a random direction may be harmful for unrelated reasons. An equally beneficial random-direction outcome would weaken a specific-direction explanation; failure of the control alone would not prove repetition semantics. Source reports an earlier orthogonal random control, so this is a different implementation of an established control idea, not a newly discovered principle.

## One new minimal intervention: reverse the published delta

W_reverse=BF16(W_base−D) on the same selected columns at both layers. This is a sign test of the final sparse patch, not the source's already reported one-layer ablations. Geometry alone motivates it: preserve sites and approximate magnitude while reversing the hypothesized causal orientation. The selected projections become approximately +7.00057/+7.00038 of their originals; realized update norm ratios are 1.000045/1.000038. This large opposite dose can harm general function, so deterioration alone does not establish semantic specificity.

Freeze paired prompts, generation settings, token cap, task accuracy and repetition measures before generating either intervention. Use original/published/permuted/reverse conditions on the same prompts. Do not redefine success from observed truncations. Similar improvement from both signs would falsify the simple orientation-specific suppression account on that tested distribution; a uniquely beneficial published direction would be compatible with, but not prove, the account. One short or underpowered run is descriptive only. No new intervention beyond this one is proposed.

## Implementation contract

In the raw archive, layer{L}_columns are column indices; layer{L}_control_delta and layer{L}_reverse_delta are shape 1024x64 exact realized deltas from baseline. Add the selected delta to the baseline selected columns, then cast to BF16. Do not add to an already patched model. The analysis writes neither a full model nor any private user data. The behavioral worker received the control specification and arrays before this worker observed any behavior. No commits were made; parent owns final integration.

### Coordination note

The behavioral worker independently froze the same row-permutation control concept before generation, using seeds 20261107 + manifest index (0, 1), rather than this analysis's 20261107 + layer number (15, 19). It will retain its own already-frozen control; our raw control arrays are not the behavioral experiment's input. Its experiment has three arms only. The sign-reversal intervention above is an unexecuted proposal for a possible later test, not a fourth arm added to that protocol. Compare behavioral results only with that worker's own control audit and protocol.
