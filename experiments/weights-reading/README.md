# Weights-reading experiment (Claude)

One bounded, falsifiable test of whether the assistant in this session, given only the weights and architecture of Qwen3.5-0.8B, can predict which full-attention heads attend to the previous token. Result: **contradicted**; see `RESULTS.md`. The reader is this session's assistant, not the published editor model.

Read in order: `PROTOCOL.md` (frozen, sha256 `ccb9d10625c2c661…`; see the erratum in `RESULTS.md`), `PREDICTIONS.json` (frozen, sha256 `a5705fe4c1c84fd7…`), `RESULTS.md`, `RESULTS.json`.

## Files

| File | Role |
| --- | --- |
| `wsr.py` | Reimplementation of the Qwen3.5 query/key path (norms, gate split, rotary, key-head sharing) used for weights-only reading |
| `test_reading.py` | Verifies `wsr.py` against the installed library's rotary code and a randomly initialized `Qwen3_5Attention` module |
| `read_weights.py` | First reading pass: stand-in positional profiles from embedding rows; writes `predicted-profiles.json`, preserved here as `predicted-profiles-v1-flat.json` |
| `diagnose_weights.py` | Second reading pass: rotary and high-frequency energy shares per head; writes `weights-diagnostics.json` |
| `freeze_predictions.py` | Combines the two weights-only signals into the frozen `PREDICTIONS.json`, including the fixed synthetic validation set |
| `PREDICTIONS-v1-draft-not-frozen.json` | First-pass draft, preserved; its hash `1b4b1dac…` is the recorded pre-freeze draft hash |
| `analysis.py`, `test_analysis.py` | Measurement and statistics helpers (m1, profiles, Spearman, permutation null) with tests |
| `validate.py` | The probe: refuses to run unless `PREDICTIONS.json` matches the logged hash; hooks the six attention modules; writes `RESULTS.json` |
| `RESULTS.json`, `RESULTS.md` | Aggregate measurements, verdicts, interpretation |

Raw per-sequence statistics and run manifests live in ignored `private/weights-reading/run-*/`. Synthetic passages, frozen token IDs and aggregate statistics are versioned. No private email, full activation tensors or model weights are stored in Git.

## Evidence inventory: what the reader used

Weights: `embed_tokens.weight` rows (stand-in inputs and an input second-moment matrix); `input_layernorm`, `q_proj`, `k_proj`, `q_norm`, `k_norm` of layers 3, 7, 11, 15, 19, 23. Architecture: `config.json` and the installed Transformers modeling code. Tokenizer: only to encode the reader's own English passages for set B. Not used before the freeze: any model output, activation on text, label or probe. Public prior knowledge disclosed in `PROTOCOL.md`.

## Run

The reading/freezing scripts overwrite their output files. Do not rerun them in this frozen experiment directory. For a new experiment, first preserve the existing commit and archived run, then work in a separate copy and record a new protocol. The commands below document the original workflow, not a safe in-place resume procedure.

```sh
cd <repo root>
OMP_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1 private/tiny-replay/venv/bin/python -m unittest discover -s experiments/weights-reading -v
PYTHONDONTWRITEBYTECODE=1 private/tiny-replay/venv/bin/python experiments/weights-reading/read_weights.py        # reading pass 1 (CPU)
PYTHONDONTWRITEBYTECODE=1 private/tiny-replay/venv/bin/python experiments/weights-reading/diagnose_weights.py    # reading pass 2 (CPU)
cd experiments/weights-reading && PYTHONDONTWRITEBYTECODE=1 ../../private/tiny-replay/venv/bin/python freeze_predictions.py
PYTHONDONTWRITEBYTECODE=1 ../../private/tiny-replay/venv/bin/python validate.py --predictions-sha256 <logged hash> --device mps
```

Re-running `freeze_predictions.py` rewrites `PREDICTIONS.json` with a new timestamp; the logged hash in the local experiment record (not distributed) is the record of what was frozen.
