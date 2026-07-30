# Convenience targets. Run `make setup` first, then `make smoke` / `make baseline`.
VENV := .venv
PY   := $(VENV)/bin/python
PIP  := $(VENV)/bin/pip

# Override to evaluate a different adapter, e.g.
#   make eval-all ADAPTER=adapters/lora-qwen2.5-0.5b-aug
ADAPTER ?= adapters/lora-qwen2.5-0.5b

.PHONY: help setup dev-setup freeze smoke baseline data train eval-ft eval-ood eval-schema eval-all test lint clean

help:
	@echo "make setup     - create venv and install local (CPU/MPS) requirements"
	@echo "make smoke     - fast end-to-end pipeline check (tiny model, 5 examples)"
	@echo "make baseline  - run the real baseline eval (Qwen 0.5B Instruct)"
	@echo "make data      - (re)generate the de-leaked NL->SQL training set"
	@echo "make train     - LoRA fine-tune the base model on data/train/"
	@echo "make eval-ft   - evaluate the fine-tuned adapter on the eval set"
	@echo "make eval-ood  - evaluate the adapter on the out-of-template (reworded) eval set"
	@echo "make eval-schema - evaluate base + adapter on the second (bookstore) schema"
	@echo "make eval-all  - evaluate the adapter on all three eval sets (regression check)"
	@echo "make test      - run the fast unit tests (no model download)"
	@echo "make lint      - run ruff (real-error rules) over the repo"
	@echo "make freeze    - pin installed versions into requirements.txt"

setup:
	python3 -m venv $(VENV)
	$(PIP) install --upgrade pip setuptools wheel
	$(PIP) install -r requirements.in
	$(PIP) freeze > requirements.txt
	@echo "Done. Activate with: source $(VENV)/bin/activate"

freeze:
	$(PIP) freeze > requirements.txt

dev-setup:
	$(PIP) install -r requirements-dev.txt

test:
	$(PY) -m pytest

lint:
	$(PY) -m ruff check .

smoke:
	$(PY) -m src.eval_baseline --smoke --limit 5

baseline:
	$(PY) -m src.eval_baseline

data:
	$(PY) -m src.build_dataset

train:
	$(PY) -m src.train_lora

eval-ft:
	$(PY) -m src.eval_baseline --adapter $(ADAPTER)

eval-ood:
	$(PY) -m src.eval_baseline --adapter $(ADAPTER) \
		--eval-file data/eval/text2sql_eval_paraphrase.jsonl

eval-schema:
	$(PY) -m src.eval_baseline \
		--eval-file data/eval/text2sql_eval_bookstore.jsonl --schema bookstore
	$(PY) -m src.eval_baseline --adapter $(ADAPTER) \
		--eval-file data/eval/text2sql_eval_bookstore.jsonl --schema bookstore

# Run the adapter over every eval set: in-template, reworded, and unseen schema.
# Use this after retraining to check a gain on one set is not a loss on another.
eval-all:
	$(PY) -m src.eval_baseline --adapter $(ADAPTER)
	$(PY) -m src.eval_baseline --adapter $(ADAPTER) \
		--eval-file data/eval/text2sql_eval_paraphrase.jsonl
	$(PY) -m src.eval_baseline --adapter $(ADAPTER) \
		--eval-file data/eval/text2sql_eval_bookstore.jsonl --schema bookstore

clean:
	rm -rf $(VENV) src/__pycache__ **/__pycache__
