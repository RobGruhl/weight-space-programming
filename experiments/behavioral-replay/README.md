# Local bounded behavioral replay

This isolated experiment extends the earlier inconclusive tiny smoke with a frozen three-arm paired comparison. Read `PROTOCOL.md` first. The exact p50c patch and one predeclared shuffled-direction negative control are applied to internal MLP weights of Qwen3.5-0.8B, not email classification or the output head. The model checkpoint remains unchanged on disk.

Run from the repository root with the existing isolated environment; no installs are needed:

```sh
PYTHONDONTWRITEBYTECODE=1 private/tiny-replay/venv/bin/python -m unittest discover -s experiments/behavioral-replay -v
PYTHONDONTWRITEBYTECODE=1 private/tiny-replay/venv/bin/python experiments/behavioral-replay/run.py --prepare
```

Preparation hashes the checkpoint and reads only the two edited tensors on CPU. It does not construct a model or allocate on MPS. The synthetic tests import the runtime but do not load weights or initialize a model.

Only after the operator confirms local GPU resources are available:

```sh
PYTHONDONTWRITEBYTECODE=1 private/tiny-replay/venv/bin/python -u experiments/behavioral-replay/run.py --gpu-released
```

Every invocation creates a new timestamped directory under ignored `private/behavioral-replay/`. There is no resume/overwrite mode: preserve and report partial attempts before deciding whether another run is justified. The flag is a coordination guard, not a substitute for permission. A sampled memory limit and wall-clock watchdog protect resources but cannot guarantee immediate termination inside a blocked native call.

The source model revision, upstream provenance and license are documented in `../tiny-replay/README.md`. No vendored upstream scripts are imported or executed. `run.py` adapts our own earlier local replay implementation. `metrics.py` separates completed answer grading, format compliance, censoring and repetition. A manual semantic review may supplement automated box parsing but must be explicitly recorded, never silently overwrite the raw fields.

The completed twelve-sample comparison is development only. Four synthetic questions and one generation seed cannot measure benchmark accuracy or a general edit effect. See `RESULTS.md` for the one baseline-relative completion rescue, control limitations and termination-policy mismatch. `analyze.py` and `audit_termination.py` are separate post-hoc analyses; they never modify the frozen run or raw result fields. No further GPU generation is planned.
