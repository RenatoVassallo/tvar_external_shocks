# Full replication: `make all` (about 10 minutes). Needs the environment of pyproject.toml.
PY ?= python

.PHONY: all quick data test figures tables clean

all:
	$(PY) -m tvar_shocks all

quick:            ## checks that everything runs; the numbers are not the paper's
	$(PY) -m tvar_shocks all --quick

data:             ## rebuilds the database from the sources and compares it with the frozen copy
	$(PY) -m tvar_shocks data

figures:
	$(PY) -m tvar_shocks figures

tables:
	$(PY) -m tvar_shocks tables

test:
	$(PY) -m pytest -q

clean:
	rm -rf output/results output/figures output/tables
