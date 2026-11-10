# Patch audit (Claude)

Independent, read-only audit of the upstream p50c weight edit against a local copy of `Qwen/Qwen3.5-0.8B` at revision `2fc06364715b967f1860aea9cf38778875588b17`. Standard-library Python only (`struct`, `zipfile`, `hashlib`, `array`): no numpy, torch, safetensors or upstream code, so it shares no implementation with upstream's `apply_edit.py` or the tiny-replay adapter.

```sh
python3 -I experiments/patch-audit/selftest_audit.py                                   # synthetic fixture: must print SELFTEST PASS
python3 -I experiments/patch-audit/audit_p50c_stdlib.py private/tiny-replay/model/model.safetensors references/upstream/exp8_math/runs/p50c/edit /tmp/report.json
```

`p50c-audit-report-2026-11-07.json` is the report from the first run: hashes and counts only, no model content. All checks passed and matched the upstream manifest, the separate adapter's reconstruction and integrity reports (not distributed).

This establishes that the local file is the published checkpoint and that the patch rebuilds the evaluated tensors bit for bit. It says nothing about model behavior.

## Additional independent audits

- `geometry_stdlib.py` and `p50c-geometry-2026-11-07.json` use direct BF16 parsing and power iteration to check the two-tensor edit geometry without loading the model. They independently confirm an approximately rank-one delta with fitted alpha 6.000321 and 6.000173. The selected directional component reverses to about minus five times its original size; this is not orthogonal projection removal. Only 2.01% and 1.08% of the selected base-column energy lay along that direction. These are geometric facts, not semantic evidence about repetition.
- `loop_diagnostics_stdlib.py` and `loop-diagnostics-2026-11-07.json` independently scan the completed behavioral replay's saved token IDs. On all 12 samples they agree with the separate block-based implementation on repeated-run maxima of at least two, all four-repeat flags and compression ratios. The implementations differ only in reporting zero versus one when no repeated block exists. Reports contain synthetic case names and counts, no decoded text or private mail.

The loop diagnostics were added after outcomes began and are exploratory. They do not replace the frozen detector or change stopping/grading. See `../behavioral-replay/RESULTS.md` for the full interpretation, control limitations and termination-token caveat. Source and report contents are pinned together by the local Git commit.
