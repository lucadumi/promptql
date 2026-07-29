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

Held-out eval: **20 examples** (`data/eval/text2sql_eval.jsonl`), scored two ways. Two further
sets probe generalisation (see below): an **out-of-template** set
(`data/eval/text2sql_eval_paraphrase.jsonl`) re-asks the same 20 intents with unfamiliar
phrasings, and a **cross-schema** set (`data/eval/text2sql_eval_bookstore.jsonl`) re-asks the
same 20 intents against a completely different (bookstore) schema.

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
The next section closes that gap.

### Cross-schema: does it transfer to a new schema?

The two evals above keep the **employees** schema fixed. To test whether the fine-tune
memorised that schema's column names or actually learned to read the schema from the prompt,
`data/eval/text2sql_eval_bookstore.jsonl` re-asks the same 20 intents against a completely
different **bookstore** schema (`publishers`, `books`, with all-new column names), built to
mirror the original construct-for-construct. Same in-template phrasing, brand-new schema and
seeded DB, so only the schema changes.

| Model                          | employees (exec) | bookstore / unseen (exec) |
|--------------------------------|:----------------:|:-------------------------:|
| `Qwen2.5-0.5B-Instruct` (base) | 65% (13/20)        | 70% (14/20)               |
| `+ LoRA fine-tune`             | 100% (20/20)       | **100% (20/20)**          |

The fine-tune holds **100% / 100%** on a schema it never trained on, and the predictions use
the bookstore tables and columns (`books`, `publishers`, `price`, `genre`) with **zero**
leakage of employees-schema names. So the LoRA did **not** just memorise employees column
names - it learned to copy the schema's tables and columns out of the prompt and slot them
into the right query shape. Put together with the phrasing result above, the brittleness is
specifically about **phrasing**, not **schema**: at this point rewording a question dropped
it to 75%, while swapping the entire schema kept it at 100%. That is what the next section
sets out to fix.

**Be honest about the "after":** the 100% is real and **leakage-free** - no eval question
or SQL appears in training, enforced in `src/build_dataset.py`. But the eval set is
*in-distribution* with the synthetic training templates (same SQL patterns, unseen literal
values), so it shows LoRA taught the model the target patterns and to copy values from the
question. Execution accuracy, the out-of-template phrasing eval, and the cross-schema eval
above now measure the generalisation gap directly.

### Fixing the weakness: paraphrase-augmented training

The three evals above localise exactly one weakness: **phrasing**. So the training data was
rewritten to attack it. Previously each SQL pattern had exactly one question wording, which
is precisely what a model overfits to. Now every pattern ships a list of interchangeable
phrasings that vary register, synonyms and sentence shape, and the generator expands each
pattern over all of them (~2 phrasings per SQL target, 316 examples, up from 176). Two
further curation rules came out of the experiment:

- **Balance the pattern mix.** Parameter pools differ wildly in size, so naive expansion made
  some patterns 14x more frequent than others (55 examples vs 4). The model started answering
  rarer patterns with a frequent pattern's shape, e.g. returning the `GROUP BY ... HAVING`
  shape for a plain `GROUP BY` question. Capping each pattern at 24 examples fixed it.
- **Avoid ambiguous wordings.** Describing `SELECT department FROM employees` as "the
  departments of all employees" taught the model that the word "departments" implies a
  `department` column, and it then answered "List all department names" with
  `SELECT department FROM departments` - a column that does not exist.

Same base model, same LoRA hyper-parameters, same three eval sets; only the training data
changed:

| Eval set                       | before augmentation | after augmentation |
|--------------------------------|:-------------------:|:------------------:|
| in-template                    | 100% (20/20)        | **100% (20/20)**   |
| out-of-template (reworded)     | 75% (15/20)         | **90% (18/20)**    |
| cross-schema (bookstore)       | 100% (20/20)        | **100% (20/20)**   |

<sub>Execution accuracy. Exact-match on the reworded set rises 70% -> 80%.</sub>

**+15 points on the weakness, with no regression on either other set.** Three of the five
original out-of-template failures are fixed: "rank every employee by pay" now projects `name`
instead of `*`, "who is our lowest-paid employee" returns the name instead of `MIN(salary)`,
and "which departments are bigger than 10 people" now emits a valid `GROUP BY ... HAVING`
instead of an invalid `WHERE COUNT(*) > 10`.

The **two remaining failures are the same mistake**: "what's our total headcount?" and "how
big is the Sales team?" both produce `SUM(salary)` instead of `COUNT(*)`. The model reads
words of magnitude ("total", "how big") as a request to add up money. That is a real,
specific limitation, and no amount of the current phrasing pool has taught it otherwise.

**The honest caveat.** These eval results guided data-curation decisions (the balancing cap
and the ambiguity fix were both diagnosed by reading eval failures), so the in-template
number is no longer a fully blind measurement. Nothing was tuned *to the answers* - the
leakage filter still removes every eval question and every eval gold SQL from training, so
patterns like `SELECT name FROM departments` have **zero** training examples and must be
reached by generalisation - but the honest framing is that the eval sets are now development
signal, and a genuinely blind estimate would need a fresh, unseen set.

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
make eval-schema # score base + adapter on the second (bookstore) schema
make eval-all   # score the adapter on all three eval sets (regression check)
```

`make data && make train` reproduces the paraphrase-augmented adapter reported above.
Point any eval target at a different adapter with `ADAPTER=`:

```bash
make eval-all ADAPTER=adapters/lora-qwen2.5-0.5b-aug
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
│   ├── text2sql_eval_bookstore.jsonl   # same 20 intents, unseen bookstore schema (cross-schema)
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
- ✅ Add a **second (bookstore) schema** to test cross-schema generalisation: the fine-tune
  holds 100% execution accuracy on a schema it never trained on, so it reads the schema from
  the prompt rather than memorising column names.
- ✅ **Close the phrasing gap**: paraphrase-augment and balance the training set, retrain, and
  re-measure. Out-of-template execution accuracy goes 75% -> 90% with no regression on the
  in-template or cross-schema sets.
- Next: teach the remaining failure (words of magnitude like "total headcount" and "how big
  is the team" still map to `SUM(salary)` instead of `COUNT(*)`), and add multi-table
  `JOIN`s, which no eval set covers yet.
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
