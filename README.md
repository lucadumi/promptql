# PromptQL - fine-tune a small LLM for natural language to SQL

> Take a small, **open** language model I fully control, teach it to translate plain-English
> questions into SQL, and **prove** it improved with an honest before/after evaluation.

**Status:** ✅ fine-tuned + evaluated two ways. A LoRA adapter (only **0.88%** of params
trained) lifts the base `Qwen2.5-0.5B-Instruct` from **40% → 100%** exact-match and
**65% → 100%** execution-accuracy on the held-out set - a real gain measured on a training
set kept strictly separate and **de-leaked** from eval.
(The eval is *in-distribution* with the synthetic training patterns, so this measures
pattern generalisation + value-copying; see the honesty caveat under Results.)

---

## What this project is

A complete **post-training + evaluation loop** in miniature:

1. Pick a small open model I can run and modify myself.
2. **Measure it first** (the "before").
3. **Fine-tune** it on a narrow task (text-to-SQL) using LoRA.
4. **Re-measure** on the *same* held-out set (the "after"), and be honest about what got
   worse, not just what got better.

It's deliberately small and cheap so the whole loop runs on a laptop + a free cloud GPU -
the point is the *method*, not the model size.

---

## The task: text-to-SQL

Given a **fixed database schema** and a question in English, the model must output a
single SQL query. The schema is shown to the model in every prompt:

```sql
CREATE TABLE departments (id INTEGER PRIMARY KEY, name TEXT, budget INTEGER, location TEXT);
CREATE TABLE employees  (id INTEGER PRIMARY KEY, name TEXT, department TEXT,
                         salary INTEGER, hire_date TEXT, manager_id INTEGER);
```

| Input (question)                 | Target (SQL)                          |
|----------------------------------|---------------------------------------|
| "How many employees are there?"  | `SELECT COUNT(*) FROM employees`      |
| "List all department names."     | `SELECT name FROM departments`        |

---

## Results

Held-out eval: **20 examples** (`data/eval/text2sql_eval.jsonl`), scored two ways. A second,
**out-of-template** set (`data/eval/text2sql_eval_paraphrase.jsonl`) re-asks the same 20
intents with unfamiliar phrasings to test generalisation (see below).

| Model                                    | Params        | exact-match      | exec-accuracy    |
|------------------------------------------|:-------------:|:----------------:|:----------------:|
| `Qwen2.5-0.5B-Instruct` (base, no FT)    | 0.49B         | **40% (8/20)**   | **65% (13/20)**  |
| `+ LoRA fine-tune` (r=8)                 | +4.4M (0.88%) | **100% (20/20)** | **100% (20/20)** |

<sub>Greedy (deterministic) decoding, run on Apple Silicon GPU (MPS) in ~7-10s.
Full per-example predictions in `results/`.</sub>

**Read the two scores.** *Exact-match* is strict normalized string equality - a
semantically-correct query that differs by a single keyword (e.g. `ORDER BY salary` vs
`ORDER BY salary ASC`) is counted **wrong**. *Execution accuracy* runs the predicted and
gold SQL against a real **seeded SQLite DB** (`src/db.py`) and compares the returned rows,
so it credits correct-but-differently-written queries. The gap between them is exactly
those queries: the base model actually answers **13/20** correctly, it just phrases 5 of
them unlike the gold string - e.g. `COUNT(id)` vs `COUNT(*)`, `SELECT DISTINCT name` vs
`SELECT name`, `ORDER BY salary` vs `ORDER BY salary ASC`. So exact-match (40%) is a
conservative lower bound; execution accuracy (65%) is the honest "before".

### Out-of-template phrasing: does it generalise?

The table above is *in-distribution* with the training templates, so 100% mostly proves
pattern-fit. To probe generalisation, `data/eval/text2sql_eval_paraphrase.jsonl` keeps the
**same 20 gold queries** but rewrites every question in unfamiliar, indirect language
(verified to overlap no training question). Same schema, same seeded DB, so only the wording
changes.

| Model                          | in-template (exec) | out-of-template (exec) |
|--------------------------------|:------------------:|:----------------------:|
| `Qwen2.5-0.5B-Instruct` (base) | 65% (13/20)        | 55% (11/20)            |
| `+ LoRA fine-tune`             | 100% (20/20)       | **75% (15/20)**        |

The fine-tune falls from **100% to 75%** execution accuracy (and 100% to 70% exact-match)
once the questions are reworded: LoRA learned the template patterns strongly but only
*partly* generalises to new phrasings. It still clears the base model on the same OOD set
(75% vs 55%), so it learned intent, not just surface strings. The misses are telling: "our
headcount" and "how big is the Sales team" get `SUM(salary)` instead of `COUNT(*)`, the
"lowest-paid employee" comes back as `MIN(salary)` instead of the name, and "bigger than 10
people" emits an invalid `WHERE COUNT(*) > 10` instead of `GROUP BY ... HAVING`. That is the
honest ceiling of a narrow LoRA fine-tune, and exactly why an out-of-template set matters.

**Be honest about the "after":** the 100% is real and **leakage-free** - no eval question
or SQL appears in training, enforced in `src/build_dataset.py`. But the eval set is
*in-distribution* with the synthetic training templates (same SQL patterns, unseen literal
values), so it shows LoRA taught the model the target patterns and to copy values from the
question, **not** robustness to out-of-distribution phrasings or new schemas. Execution
accuracy and the out-of-template phrasing eval above now measure that gap directly; the
remaining honest next step is a *second schema* (cross-schema generalisation).

---

## Quickstart

**Requirements:** Python 3.9+, macOS or Linux. On Apple Silicon the Mac GPU (MPS) is used
automatically; no NVIDIA card needed to *run* the baseline.

```bash
make setup      # create a venv and install torch / transformers / datasets / peft / accelerate
make baseline   # reproduce the 40% baseline (downloads Qwen2.5-0.5B-Instruct once)
make smoke      # fast end-to-end pipeline check on a tiny model (no big download)
make data       # (re)generate the de-leaked NL->SQL training set into data/train/
make train      # LoRA fine-tune on data/train/ -> adapters/ (uses MPS / CPU / CUDA)
make eval-ft    # score the fine-tuned adapter on the same eval set (the "after")
make eval-ood   # score the adapter on the out-of-template (reworded) eval set
```

Try any other small model:

```bash
python -m src.eval_baseline --model Qwen/Qwen2.5-1.5B-Instruct
python -m src.eval_baseline --model HuggingFaceTB/SmolLM2-360M-Instruct --limit 10
```

---

## How it works (method)

- **Prompt format** - schema + question, rendered with the model's own chat template.
  The *exact same* format is used for training and evaluation to avoid prompt-format
  drift (a classic silent-failure bug). → `src/data_utils.py`
- **Generation** - greedy / deterministic (`do_sample=False`), so every run is
  reproducible and before/after comparisons are fair. → `src/eval_baseline.py`
- **Scoring (two metrics)** - *exact-match*: normalize both prediction and gold (strip
  markdown fences, lowercase, collapse whitespace, unify quotes, drop trailing `;`) and
  compare as strings → `src/metrics.py`. *Execution accuracy*: run both queries against a
  small **seeded SQLite DB** and compare the returned rows - order-sensitive only when the
  gold query uses `ORDER BY` → `src/db.py`.
- **Seed database** - a deterministic, committed dataset (`src/db.py`) for the fixed
  schema, hand-chosen so every eval query returns a *discriminating* result (distinct
  salaries; one department with >10 employees; budgets, locations and hire-dates that
  straddle the eval thresholds). A wrong query therefore returns different rows.
- **Output** - a machine-readable `results/*.json` (every prediction, both metrics) plus a
  human-readable `results/baseline.md` leaderboard.

---

## Repo layout

```
.
├── src/
│   ├── eval_baseline.py   # run a model on the eval set, save the before/after numbers
│   ├── build_dataset.py   # synthesise the de-leaked NL->SQL training set
│   ├── train_lora.py      # LoRA fine-tune the base model on data/train/
│   ├── data_utils.py      # DB schema + shared prompt format (keeps train == eval)
│   ├── db.py              # seed SQLite DB + execution-accuracy scoring
│   └── metrics.py         # SQL normalisation + exact-match scoring
├── data/eval/
│   ├── text2sql_eval.jsonl             # 20 held-out NL→SQL examples (in-template)
│   ├── text2sql_eval_paraphrase.jsonl  # same 20 intents, reworded (out-of-template)
│   └── README.md                       # data card: schema + how the sets were built
├── results/
│   ├── baseline.md          # human-readable results table
│   └── baseline_*.json      # full per-example predictions (committed)
├── requirements.in          # local (CPU/MPS) top-level deps
├── requirements.txt         # pinned lockfile (pip freeze)
├── requirements-gpu.txt     # CUDA-only extras (bitsandbytes) for the GPU box
└── Makefile                 # make setup / smoke / baseline
```

---

## Roadmap

- ✅ Curate a dedicated NL→SQL **training** set, kept strictly separate from the eval set.
- ✅ **Fine-tune** the base model with LoRA.
- ✅ Add **execution-accuracy** evaluation (run predicted vs. gold SQL on a real seeded
  SQLite DB), crediting correct-but-differently-written queries.
- ✅ Add an **out-of-template phrasing** eval (same intents, unfamiliar wording): the
  fine-tune holds 75% execution accuracy vs 100% in-template, quantifying the gap.
- Next: a **second schema** to test cross-schema generalisation.
- Optional: quantize the model and serve it behind a small API.

---

## Compute & environment

- **Local (this repo):** Apple Silicon macOS, CPU/MPS - used for scaffolding and light
  evaluation.
- **Training:** a cloud **T4** (Google Colab / Kaggle, free tier). `bitsandbytes` / 4-bit
  QLoRA is **CUDA-only**, so it lives in `requirements-gpu.txt` and is *not* installed
  locally.

## Tech stack

Python · PyTorch · Hugging Face `transformers`, `datasets`, `peft` (LoRA), `accelerate`
(+ `bitsandbytes` on the GPU). Experiment tracking: JSON/CSV now, Weights & Biases optional.

## Scope (tracks)

- **Track A - LoRA fine-tune a small open model (current).** Post-training + evaluation on
  a focused task. Cheap, fits a free GPU.
- **Track B - pre-train a tiny GPT from scratch (stretch, nanoGPT-style).** To exercise
  the pre-training loop itself on a compact corpus.

---

## Evaluation principles

- Measure a baseline **before** training - otherwise improvement can't be proven. ✅ done.
- Watch **eval** loss, not just train loss (a tiny dataset overfits fast).
- Keep tokenization / prompt format **identical** between training and evaluation.
- Always report on the held-out eval set; no results claimed without it.

## Why this project

An end-to-end demonstration of small-LLM post-training and honest evaluation: data
curation, LoRA fine-tuning, GPU training (including memory/OOM troubleshooting), and a
rigorous before/after comparison on a held-out set - rather than calling a hosted API.
