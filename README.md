# Weight-space programming: a small replication lab

Can an LLM inspect neural-network weights, predict behavior, and make useful surgical edits? This repository records bounded local experiments inspired by [Sparks of Weight-Space Programming](https://rikiyat.github.io/blog/weight-space-programming/) and the authors’ [research artifacts](https://github.com/RikiyaT/lm_understand-logs).

**[Read the illustrated field guide](https://robgruhl.github.io/weight-space-programming/)**

## What we learned

- **Patch reconstruction works.** An independent standard-library audit reconstructed the published Qwen3.5-0.8B p50c tensors bit for bit. The patch changed 130,188 BF16 entries in two internal MLP matrices.
- **Small behavioral replay: a limited positive result.** On four reused toy arithmetic questions, one seed and an 8,192-token cap, baseline completed 3/4 correctly, p50c 4/4, and a shuffled-direction control 3/4. This is not a benchmark, a general improvement claim, or discovery of a new patch. The stopping-token mismatch and control limitations are documented.
- **Weights-only head reading: a negative result.** Frozen rankings over 48 attention heads failed on 20 synthetic sequences (Spearman rho −0.056). The layer prediction failed; the top-versus-comparator test was inconclusive. This does not disprove weights reading generally.
- **Closed-form head edits are algebraically possible.** A synthetic rank-one construction changes a target margin while preserving chosen controls. These controls are construction constraints, not held-out evidence of semantic improvement.

## Start here

| Experiment | Evidence |
|---|---|
| [Bounded behavioral replay](experiments/behavioral-replay/RESULTS.md) | Three arms, paired outputs, censoring, repetition diagnostics and termination caveat |
| [Weights-only reading](experiments/weights-reading/RESULTS.md) | Frozen predictions, synthetic validation and failed hypotheses |
| [Independent patch audit](experiments/patch-audit/README.md) | Raw-byte reconstruction and independent geometry checks |
| [Weight geometry](experiments/weight-structure/README.md) | Approximately rank-one edit, directional overshoot, limitations |
| [Tiny smoke replay](experiments/tiny-replay/RESULTS.md) | Earlier short-token run, all outputs truncated |
| [Synthetic direct edit](experiments/head-algebra/direct_patch.py) | Closed-form algebra, no optimizer or model inference |

## Safe quick checks

Python’s standard library is enough for the algebra and raw-byte audit self-tests:

```sh
make test
python3 experiments/head-algebra/direct_patch.py
python3 -m wsp.weight_edit
```

These do not download weights, contact APIs or run a model. The numerical test suites require the documented PyTorch/Transformers environment; see [reproduction setup](REPRODUCING.md). No API credentials or paid services are required. The runtime scripts use local model files with remote-code trust disabled.

## Scope, privacy and evidence

This is a **curated public export with fresh history**, not the original working directory. It excludes personal email data and email-derived results, private coordination, credentials, machine-specific notes, model weights and raw runtime directories. [Publication notes](PUBLICATION.md) explain privacy edits and what historical hashes refer to. Synthetic passages and aggregate results are included. Archived raw traces and full timestamp evidence are not distributed, limiting external verification of historical runs.

Do not confuse replaying a published patch with independently reasoning from weights to discover it. No new edit-discovery or production decision-quality improvement is established here. The source checkpoint is Qwen3.5-0.8B; these results do not establish improvement to other models.

## Attribution and licensing

The research source is [RikiyaT/lm_understand-logs](https://github.com/RikiyaT/lm_understand-logs), pinned at `4d09ba6d6a7e891dd8063974dfe81b3d78e3494f`. Its licensing distinguishes MIT scripts, CC BY 4.0 logs/data/networks, and separately licensed quoted content. See [upstream license](references/UPSTREAM-LICENSE) and [reference setup](references/README.md). This repository does not relicense upstream content or redistribute model weights. No blanket license has been added for this project’s original material.
