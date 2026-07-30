# Convenience targets. Run `make setup` first, then `make smoke` / `make baseline`.
VENV := .venv
PY   := $(VENV)/bin/python
PIP  := $(VENV)/bin/pip

# Override to evaluate a different adapter, e.g.
#   make eval-all ADAPTER=adapters/lora-qwen2.5-0.5b-aug
# `make train` writes here too, so train + eval stay in sync by default.
ADAPTER ?= adapters/lora-qwen2.5-0.5b-join

# Extra flags for `make train`. On a memory-constrained machine, a smaller micro
# batch with matching accumulation keeps the effective batch (8) identical:
#   make train TRAIN_ARGS="--batch-size 4 --grad-accum 2"
TRAIN_ARGS ?=

.PHONY: help setup dev-setup freeze smoke baseline data train eval-ft eval-ood eval-schema eval-join eval-all eval-blind test lint clean

help:
	@echo "make setup     - create venv and install local (CPU/MPS) requirements"
	@echo "make smoke     - fast end-to-end pipeline check (tiny model, 5 examples)"
	@echo "make baseline  - run the real baseline eval (Qwen 0.5B Instruct)"
	@echo "make data      - (re)generate the de-leaked NL->SQL training set"
	@echo "make train     - LoRA fine-tune the base model on data/train/"
	@echo "make eval-ft   - evaluate the fine-tuned adapter on the eval set"
	@echo "make eval-ood  - evaluate the adapter on the out-of-template (reworded) eval set"
	@echo "make eval-schema - evaluate base + adapter on the second (bookstore) schema"
	@echo "make eval-join - evaluate base + adapter on the multi-table JOIN eval sets"
	@echo "make eval-all  - evaluate the adapter on all five DEV eval sets (regression check)"
	@echo "make eval-blind  - HELD-BACK set: score base + adapter ONCE, then stop (see README)"
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
	$(PY) -m src.train_lora --output-dir $(ADAPTER) $(TRAIN_ARGS)

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

# The multi-table JOIN sets: the employees schema joins on a TEXT key, the
# bookstore mirror joins on an INTEGER foreign key the training data never shows.
eval-join:
	$(PY) -m src.eval_baseline --adapter $(ADAPTER) \
		--eval-file data/eval/text2sql_eval_join.jsonl
	$(PY) -m src.eval_baseline --adapter $(ADAPTER) \
		--eval-file data/eval/text2sql_eval_join_bookstore.jsonl --schema bookstore

# Run the adapter over every DEVELOPMENT eval set: in-template, reworded, unseen
# schema, and the two JOIN sets. Use this after retraining to check a gain on one
# set is not a loss on another.
#
# `eval-blind` is deliberately NOT part of this target. Every set below has fed at
# least one data-curation decision, which is what makes them development signal;
# the blind set only keeps its meaning while it stays out of that loop.
eval-all:
	$(PY) -m src.eval_baseline --adapter $(ADAPTER)
	$(PY) -m src.eval_baseline --adapter $(ADAPTER) \
		--eval-file data/eval/text2sql_eval_paraphrase.jsonl
	$(PY) -m src.eval_baseline --adapter $(ADAPTER) \
		--eval-file data/eval/text2sql_eval_bookstore.jsonl --schema bookstore
	$(PY) -m src.eval_baseline --adapter $(ADAPTER) \
		--eval-file data/eval/text2sql_eval_join.jsonl
	$(PY) -m src.eval_baseline --adapter $(ADAPTER) \
		--eval-file data/eval/text2sql_eval_join_bookstore.jsonl --schema bookstore

# The held-back set. Written after the model was frozen, never used to diagnose a
# failure or steer the training data, and excluded from `eval-all` so it cannot
# quietly become development signal. Protocol: score a given model ONCE, record
# the number, and do not curate against it. Reading a failure here and "fixing"
# it converts this into just another dev set, and the project loses the only
# unbiased estimate it has.
eval-blind:
	$(PY) -m src.eval_baseline --eval-file data/eval/text2sql_eval_blind.jsonl
	$(PY) -m src.eval_baseline --adapter $(ADAPTER) \
		--eval-file data/eval/text2sql_eval_blind.jsonl

clean:
	rm -rf $(VENV) src/__pycache__ **/__pycache__
