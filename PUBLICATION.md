# Public release boundaries

This is a curated export with fresh Git history, not a mirror of the original working directory. It contains synthetic experiments and aggregate research results. Personal email data, email-derived results, the mailbox workbench, private coordination logs, credentials, machine-specific notes, model weights, local environments and raw runtime directories are excluded. The original working copy is retained separately.

Some Markdown and code comments were edited to remove private coordination details and make operator instructions generic. In particular, the weights-reading protocol is a privacy-edited copy. Its historical SHA-256 values refer to the original private freeze, not the edited public Markdown. Prediction/result JSON remains unchanged; archived raw run logs and full pre-run timestamp evidence are not distributed. Do not treat a fresh public commit as an independently timestamped preregistration.

Modified existing files:
- `experiments/behavioral-replay/run.py`
- `experiments/behavioral-replay/README.md`
- `experiments/behavioral-replay/PROTOCOL.md`
- `experiments/weights-reading/README.md`
- `experiments/weights-reading/RESULTS.md`
- `experiments/weights-reading/PROTOCOL.md`
- `experiments/patch-audit/README.md`
- `experiments/tiny-replay/README.md`
- `experiments/behavioral-replay/RESULTS.md` (public provenance clarification)
- `wsp/__init__.py` (package description)
- `experiments/patch-audit/selftest_audit.py` (temporary artifact cleanup and failing exit status)

No new model experiment was run for publication.
