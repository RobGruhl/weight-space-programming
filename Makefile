PYTHON ?= python3
.PHONY: test
test:
	PYTHONDONTWRITEBYTECODE=1 $(PYTHON) -m unittest discover -s tests -v
	PYTHONDONTWRITEBYTECODE=1 $(PYTHON) -I experiments/patch-audit/selftest_audit.py
