# Convenience targets. Run `make setup` first, then `make smoke` / `make baseline`.
VENV := .venv
PY   := $(VENV)/bin/python
PIP  := $(VENV)/bin/pip

# Override to evaluate a different adapter, e.g.
#   make eval-all ADAPTER=adapters/lora-qwen2.5-0.5b-aug
# `make train` writes here too, so train + eval stay in sync by default.
ADAPTER ?= adapters/lora-qwen2.5-0.5b-joingroup

# Extra flags for `make train`. On a memory-constrained machine, a smaller micro
# batch with matching accumulation keeps the effective batch (8) identical:
#   make train TRAIN_ARGS="--batch-size 4 --grad-accum 2"
TRAIN_ARGS ?=

# Extra flags for `make serve`, e.g. SERVE_ARGS="--quantize --port 9000".
SERVE_ARGS ?=

# Extra flags passed to every eval in `eval-all` / `eval-blind`, e.g.
#   make eval-all EVAL_ARGS="--repair 2"
EVAL_ARGS ?=

.PHONY: help setup dev-setup freeze smoke baseline data train eval-ft eval-ood eval-schema eval-join eval-all eval-blind compare quantized repair serve test lint clean

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
	@echo "make eval-all  - evaluate the adapter on all seven DEV eval sets (regression check)"
	@echo "make eval-blind  - HELD-BACK set: score base + adapter ONCE, then stop (see README)"
	@echo "make compare   - score a 3x larger model zero-shot on every set (is the FT worth it?)"
	@echo "make quantized - score the int8-quantized adapter (accuracy cost of quantizing)"
	@echo "make repair    - score every dev set with the execute-and-repair loop on"
	@echo "make serve     - serve the adapter over HTTP on :8000 (POST /sql)"
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
# schema, the two JOIN sets, and the retired first blind set. Use this after
# retraining to check a gain on one set is not a loss on another.
#
# The *current* blind set (v2) is deliberately NOT part of this target. Every set
# below has fed at least one data-curation decision, which is what makes them
# development signal; a blind set only keeps its meaning while it stays out of
# that loop.
#
# Both *_retired.jsonl sets are here because they have been spent. Each was scored
# once as a blind set, then its failures were read and acted on -- v1 motivated the
# construct families, v2 exposed the join-grouping starvation and the vocabulary
# gap. Reading a blind set's failures is exactly what turns it into a development
# set, so they are accounted for as development sets rather than quietly re-scored
# as if still blind.
eval-all:
	$(PY) -m src.eval_baseline --adapter $(ADAPTER) $(EVAL_ARGS)
	$(PY) -m src.eval_baseline --adapter $(ADAPTER) \
		--eval-file data/eval/text2sql_eval_paraphrase.jsonl $(EVAL_ARGS)
	$(PY) -m src.eval_baseline --adapter $(ADAPTER) \
		--eval-file data/eval/text2sql_eval_bookstore.jsonl --schema bookstore $(EVAL_ARGS)
	$(PY) -m src.eval_baseline --adapter $(ADAPTER) \
		--eval-file data/eval/text2sql_eval_join.jsonl $(EVAL_ARGS)
	$(PY) -m src.eval_baseline --adapter $(ADAPTER) \
		--eval-file data/eval/text2sql_eval_join_bookstore.jsonl --schema bookstore $(EVAL_ARGS)
	$(PY) -m src.eval_baseline --adapter $(ADAPTER) \
		--eval-file data/eval/text2sql_eval_blind_v1_retired.jsonl $(EVAL_ARGS)
	$(PY) -m src.eval_baseline --adapter $(ADAPTER) \
		--eval-file data/eval/text2sql_eval_blind_v2_retired.jsonl $(EVAL_ARGS)

# The held-back set. Written after the model was frozen, never used to diagnose a
# failure or steer the training data, and excluded from `eval-all` so it cannot
# quietly become development signal. Protocol: score a given model ONCE, record
# the number, and do not curate against it. Reading a failure here and "fixing"
# it converts this into just another dev set, and the project loses the only
# unbiased estimate it has.
#
# v3 was authored by an independent party that was walled off from the training
# generator, from every existing eval set and from all model output - see the
# README section "Three blind sets, and what their disagreement measures".
eval-blind:
	$(PY) -m src.eval_baseline --eval-file data/eval/text2sql_eval_blind_v3.jsonl $(EVAL_ARGS)
	$(PY) -m src.eval_baseline --adapter $(ADAPTER) \
		--eval-file data/eval/text2sql_eval_blind_v3.jsonl $(EVAL_ARGS)

# What does letting the model read its own SQLite error buy? Scores every
# development set with one retry allowed, so the numbers line up directly against
# `eval-all`. The loop never sees the gold and only fires on a query that already
# failed to execute, so it cannot lower a score -- see src/repair.py.
repair:
	$(MAKE) eval-all EVAL_ARGS="--repair 2"

# Is a fine-tuned small model actually worth it, or would a bigger model do? This
# scores COMPARE_MODEL zero-shot on every set, including the held-back one, so the
# adapter has to justify itself against simply using a 3x larger model.
COMPARE_MODEL ?= Qwen/Qwen2.5-1.5B-Instruct

compare:
	$(PY) -m src.eval_baseline --model $(COMPARE_MODEL)
	$(PY) -m src.eval_baseline --model $(COMPARE_MODEL) \
		--eval-file data/eval/text2sql_eval_paraphrase.jsonl
	$(PY) -m src.eval_baseline --model $(COMPARE_MODEL) \
		--eval-file data/eval/text2sql_eval_bookstore.jsonl --schema bookstore
	$(PY) -m src.eval_baseline --model $(COMPARE_MODEL) \
		--eval-file data/eval/text2sql_eval_join.jsonl
	$(PY) -m src.eval_baseline --model $(COMPARE_MODEL) \
		--eval-file data/eval/text2sql_eval_join_bookstore.jsonl --schema bookstore
	$(PY) -m src.eval_baseline --model $(COMPARE_MODEL) \
		--eval-file data/eval/text2sql_eval_blind_v1_retired.jsonl
	$(PY) -m src.eval_baseline --model $(COMPARE_MODEL) \
		--eval-file data/eval/text2sql_eval_blind_v2_retired.jsonl
	$(PY) -m src.eval_baseline --model $(COMPARE_MODEL) \
		--eval-file data/eval/text2sql_eval_blind_v3.jsonl

# What does quantization cost in accuracy? int8 is CPU-only, so this also runs the
# fp32 comparison on CPU -- comparing an int8 CPU run against an fp32 MPS run would
# confound quantization with the device.
quantized:
	$(PY) -m src.eval_baseline --adapter $(ADAPTER) --device cpu
	$(PY) -m src.eval_baseline --adapter $(ADAPTER) --device cpu --quantize
	$(PY) -m src.eval_baseline --adapter $(ADAPTER) --device cpu \
		--eval-file data/eval/text2sql_eval_blind_v3.jsonl
	$(PY) -m src.eval_baseline --adapter $(ADAPTER) --device cpu --quantize \
		--eval-file data/eval/text2sql_eval_blind_v3.jsonl

serve:
	$(PY) -m src.serve --adapter $(ADAPTER) $(SERVE_ARGS)

clean:
	rm -rf $(VENV) src/__pycache__ **/__pycache__
