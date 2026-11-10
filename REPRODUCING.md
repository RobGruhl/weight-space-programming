# Reproducing the bounded experiments

## Lightweight checks

`make test` runs synthetic algebra tests and the standard-library patch-format self-test, without a model or network access.

## Numerical tests and model experiments

The historical runtime was Python 3.13.15, torch 2.15.0 and transformers 5.22.2 on Apple M4 Pro/MPS. The observed package versions are recorded in `experiments/tiny-replay/requirements.lock.txt`. That file is an environment inventory, not a hash-locked cross-platform guarantee. Create an isolated environment and install from official package sources before running numerical tests; do not install globally.

Expected layout:

```
private/tiny-replay/venv/   # isolated Python environment, ignored
private/tiny-replay/model/  # public model checkpoint, ignored
references/upstream/       # pinned public patch checkout, ignored
```

Acquire `Qwen/Qwen3.5-0.8B` revision `2fc06364715b967f1860aea9cf38778875588b17` from its official Hugging Face repository. Preserve its model license and use that revision's `config.json`, `tokenizer.json`, `tokenizer_config.json` and `chat_template.jinja`. The checkpoint named `model.safetensors-00001-of-00001.safetensors` at source is expected locally as `model.safetensors`, with SHA-256 `04b1c301231dd422b8860db31311ab2721511346a32cb1e079c4c4e5f1fe4696`. Scripts reject mismatching model bytes. Model weights are not included here. Obtain patch files as described in `references/README.md`.

Run numerical synthetic tests, without loading the checkpoint:

```sh
PYTHONDONTWRITEBYTECODE=1 private/tiny-replay/venv/bin/python -m unittest discover -s experiments/behavioral-replay -v
OMP_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1 private/tiny-replay/venv/bin/python -m unittest discover -s experiments/weights-reading -v
```

Read each experiment's README and protocol before running its model script. Some analysis/freezing scripts overwrite their own reports. Make a separate working copy first; do not overwrite the published frozen predictions or use their historical hashes to imply a new preregistration. The public protocol copy contains privacy edits, so its bytes intentionally differ from the original protocol hash recorded in `PREDICTIONS.json`. `PREDICTIONS.json` itself is unchanged.

MPS model runs allocate substantial memory. Review memory, disk and time budgets first. Runtime guards are sampled and cannot guarantee immediate termination inside a blocked native call. CPU execution may be much slower. No model inference was rerun for this public release.

## Historical evidence limits

The public repository includes source, frozen prediction arrays, synthetic validation passages, aggregate measurements and reports. Original private runtime directories, full output traces and coordination logs are excluded. Historical counts are reported, not independently re-established by publishing these files. A new replication must preserve its own versions, input hashes, artifacts and all failed attempts.
