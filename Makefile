# eroom — developer gates. Run `make check` before committing/pushing.
#
# Two boundary gates protect the open-core/private split (see BOUNDARY.md,
# src/boundary.py):
#   - snapshots : no private VALUES (embeddings/boxes) in public data/exports/*.
#   - query     : no private read-path / frontier-query MODULES in the public tree.
# Override the interpreter with `make PYTHON=.venv/bin/python check`.

PYTHON ?= python

.PHONY: check gates snapshots query query-strict test

## check: run both boundary gates + the test suite (the pre-push gate)
check: gates test

## gates: run both public/private boundary checks
gates: snapshots query

## snapshots: fail if a committed public snapshot carries a private value
snapshots:
	$(PYTHON) -m scripts.check_public_snapshots

## query: fail if a relocated private query module reappears in the public tree
##        (pending-relocation modules are a non-fatal notice until the code moves)
query:
	$(PYTHON) -m scripts.check_query_boundary

## query-strict: also fail on pending modules (the gate to use post-relocation)
query-strict:
	$(PYTHON) -m scripts.check_query_boundary --strict

## test: run the pytest suite
test:
	$(PYTHON) -m pytest -q
