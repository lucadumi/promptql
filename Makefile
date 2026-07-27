# Convenience targets. Run `make setup` first, then `make smoke` / `make baseline`.
VENV := .venv
PY   := $(VENV)/bin/python
PIP  := $(VENV)/bin/pip

.PHONY: help setup freeze smoke baseline data train eval-ft clean

help:
	@echo "make setup     - create venv and install local (CPU/MPS) requirements"
	@echo "make smoke     - fast end-to-end pipeline check (tiny model, 5 examples)"
	@echo "make baseline  - run the real baseline eval (Qwen 0.5B Instruct)"
	@echo "make data      - (re)generate the de-leaked NL->SQL training set"
	@echo "make train     - LoRA fine-tune the base model on data/train/"
	@echo "make eval-ft   - evaluate the fine-tuned adapter on the eval set"
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

data:
	$(PY) -m src.build_dataset

train:
	$(PY) -m src.train_lora

eval-ft:
	$(PY) -m src.eval_baseline --adapter adapters/lora-qwen2.5-0.5b

clean:
	rm -rf $(VENV) src/__pycache__ **/__pycache__
