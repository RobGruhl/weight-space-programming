# Public export verification

Checked November 10, 2026:

- Standard-library synthetic algebra: 3 tests passed.
- Independent patch-format self-test: passed clean and deliberately corrupted fixtures.
- Behavioral replay helpers: 10 tests passed.
- Weights-reading helpers: 5 tests passed, including a tiny randomly initialized attention module comparison.
- All exported Python sources parse and JSON files decode.
- No model checkpoint loaded, model experiment rerun, paid service used or original working-copy file changed.

Numerical tests used Python 3.13.18, torch 2.15.0 and transformers 5.22.2 with existing installed packages. The original environment launcher had a stale interpreter link; tests were run with a compatible installed Python without changing that environment. Historical experimental runtime details remain in their reports.

Privacy review uses a positive file selection, fresh Git history, exclusions documented in PUBLICATION.md, text/metadata scans and independent review. It is not a mathematical guarantee against every possible private string. Runtime artifacts, private mail, original Git history and coordination logs are not part of this export.
