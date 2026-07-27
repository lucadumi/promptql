# Convenience targets. Run `make setup` first, then `make smoke` / `make baseline`.
VENV := .venv
PY   := $(VENV)/bin/python
PIP  := $(VENV)/bin/pip

.PHONY: help setup freeze smoke baseline clean

help:
	@echo "make setup     - create venv and install local (CPU/MPS) requirements"
	@echo "make smoke     - fast end-to-end pipeline check (tiny model, 5 examples)"
	@echo "make baseline  - run the real baseline eval (Qwen 0.5B Instruct)"
	@echo "make freeze    - pin installed versions into requirements.txt"

setup:
	python3 -m venv $(VENV)
	$(PIP) install --upgrade pip setuptools wheel
	$(PIP) install -r requirements.in
	$(PIP) freeze > requirements.txt
	@echo "Done. Activate with: source $(VENV)/bin/activate"

freeze:
	$(PIP) freeze > requirements.txt

smoke:
	$(PY) -m src.eval_baseline --smoke --limit 5

baseline:
	$(PY) -m src.eval_baseline

clean:
	rm -rf $(VENV) src/__pycache__ **/__pycache__
