PYTHON ?= python

.PHONY: install install-petbox test test-petbox compile check

install:
	$(PYTHON) -m pip install -e ".[dev]"

install-petbox:
	$(PYTHON) -m pip install -e ".[dev,petbox]"

compile:
	$(PYTHON) -m compileall -q src tests examples

test:
	$(PYTHON) -m pytest -q

test-petbox:
	$(PYTHON) -m pytest -q tests/test_petbox_models.py

check: compile test
