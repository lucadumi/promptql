# PromptQL: fine-tuning a 0.5B LLM for text-to-SQL, and measuring it honestly

[![CI](https://github.com/lucadumi/promptql/actions/workflows/ci.yml/badge.svg)](https://github.com/lucadumi/promptql/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/)

> A complete post-training loop on a small open model: measure first, fine-tune with LoRA,
> re-measure on held-out sets, and report what got **worse** alongside what got better.

**Status: complete.** The model is frozen, and its final score comes from a set written by an
independent author who never saw the training data, the other eval sets, or a single model
prediction — scored once, then left alone.

A LoRA adapter training **0.88%** of the parameters takes `Qwen2.5-0.5B-Instruct` from
**41% → 84%** execution accuracy on the seven development sets. That number is *not* the
headline, because those sets were used to make decisions. This is:

| Blind set v3 (30 questions, scored once per model) | n | base 0.5B | **0.5B + LoRA** | 1.5B zero-shot |
|---|:--:|:--:|:--:|:--:|
| **easy** | 8 | 37% | **87%** | 75% |
| **medium** | 14 | 7% | **64%** | 35% |
| **hard** (correlated subqueries, `NOT EXISTS`, derived tables) | 8 | 0% | **12%** | 12% |
| **overall** | 30 | **13%** | **57%** | 40% |

<sub>Execution accuracy against a seeded SQLite DB. Greedy decoding. Full per-example
predictions in `results/`.</sub>

**Three things that table says, and a flattering benchmark would not:**

1. **The fine-tune beats a model three times its size** on questions nobody involved in
   building it had seen — 57% vs 40% — while running on a laptop CPU. 4.4M trained parameters
   bought more than 1B extra frozen ones.
2. **The hard tier is still 1/8, and the 1.5B also scores 1/8.** Compositional SQL is where
   both stop. At this scale the ceiling is task scope, not parameter count, and no amount of
   this kind of fine-tuning writes a correlated subquery it was never shown.
3. **Three blind sets, three different numbers: 75%, 40%, 57%.** Each was written by a
   different author under the same isolation, and the spread is not noise — it is how much the
   headline depends on *who writes the exam*. That spread is the most useful thing this repo
   measured, and it is the reason no single number here is quoted without saying where it came
   from.

**Jump to:** [What this demonstrates](#what-this-project-demonstrates) ·
[The task](#the-task) · [How it is measured](#how-it-is-measured) · [Results](#results) ·
[Three blind sets](#three-blind-sets-and-what-their-disagreement-measures) ·
[Is the fine-tune worth it?](#is-the-fine-tune-worth-it-vs-a-3x-larger-model) ·
[Deploying it](#deploying-it-quantization-and-a-small-api) · [Quickstart](#quickstart) ·
[Experiment log](#experiment-log-how-it-got-here) · [Honest caveats](#honest-caveats) ·
[Repo layout](#repo-layout) · [What was built](#what-was-built-in-order)

---

## What this project demonstrates

Aimed at anyone assessing this repo as work product: what it shows, and where to verify it.

| Capability | Evidence in this repo | Code |
|---|---|---|
| **Data curation for post-training** | Templated generator, several phrasings per SQL target, contrastive pairs for known confusions, per-pattern balancing, stratified split, fully seeded | `src/build_dataset.py` |
| **Leakage control that is enforced, not claimed** | Any candidate whose normalised question *or* SQL collides with any eval gold is dropped; the check reuses the exact scoring function; word-overlap to the nearest eval question is reported every build | `src/build_dataset.py`, `tests/test_build_dataset.py` |
| **Evaluation design** | Seven sets, each isolating **one** variable (phrasing / schema / construct / join key), so a score drop is diagnostic | `data/eval/` |
| **Metric design** | Exact-match *and* execution accuracy against a seeded DB, with a hand-built seed chosen so every gold discriminates | `src/db.py`, `src/metrics.py` |
| **Understanding of evaluation bias** | **Three** blind sets: one written by me, two by independent authors under enforced isolation. Two spent and retired; their 17-point disagreement is itself reported as a finding | `tests/test_eval_blind.py` |
| **Training engineering** | Explicit PyTorch loop (no `Trainer`): LoRA on attention + MLP, prompt masking, gradient accumulation, warmup + linear decay, train/val loss watched together | `src/train_lora.py` |
| **Diagnosing regressions** | Catastrophic forgetting of `JOIN`s found and fixed; a rebalancing attempt that made things worse, reported rather than buried | [Experiment log](#experiment-log-how-it-got-here) |
| **Cost/benefit honesty** | Benchmarked against a 3x larger model on every set; int8 quantization and an execute-and-repair loop both measured, with one recommended against and the other shown to buy nothing on the held-back set | [Is it worth it?](#is-the-fine-tune-worth-it-vs-a-3x-larger-model) |
| **Shipping** | Stdlib-only HTTP API that executes its own SQL behind a read-only guard, with an optional error-feedback retry, fully testable without torch | `src/serve.py`, `src/repair.py` |
| **Engineering hygiene** | 197 unit tests, ruff, CI that runs on every push without downloading a model, pinned lockfile, Makefile for every step | `.github/workflows/ci.yml`, `Makefile` |

---

## The task

Given a **fixed schema** and an English question, output one SQL query. The schema is shown
to the model in every prompt:

```sql
CREATE TABLE departments (id INTEGER PRIMARY KEY, name TEXT, budget INTEGER, location TEXT);
CREATE TABLE employees  (id INTEGER PRIMARY KEY, name TEXT, department TEXT,
                         salary INTEGER, hire_date TEXT, manager_id INTEGER);
```

| Input (question) | Target (SQL) |
|---|---|
| "How many employees are there?" | `SELECT COUNT(*) FROM employees` |
| "List all department names." | `SELECT name FROM departments` |

`employees.department` holds a department *name*, not an id; the schema is denormalised on
purpose, and that turns out to matter a great deal
([round 5 of the experiment log](#experiment-log-how-it-got-here)).

---

## How it is measured

### Two metrics, because one of them lies

- **Exact-match:** normalise both sides (strip fences, lowercase, collapse whitespace, unify
  quotes, drop trailing `;`) and compare as strings. A conservative *lower bound*: a correct
  query differing by one keyword counts as wrong. → `src/metrics.py`
- **Execution accuracy:** run prediction and gold against a **seeded SQLite database** and
  compare returned rows; order matters only when the gold uses `ORDER BY`. This credits
  correct-but-differently-written queries. → `src/db.py`

The gap between them is exactly those queries. In-template, the base model answers **13/20**
correctly but phrases 5 unlike the gold (`COUNT(id)` vs `COUNT(*)`, `ORDER BY salary` vs
`ORDER BY salary ASC`), so exact-match says 40% and execution accuracy says 65%. **65% is the
honest "before".**

The seed database is deterministic and committed, hand-chosen so every gold returns a
*discriminating* result: all salaries distinct, exactly one department over 10 people,
budgets and hire dates straddling the eval thresholds. A wrong query returns different rows.

### Eight eval sets: seven for development, one held back

| File (`data/eval/`) | n | Schema | What it isolates |
|---|:--:|---|---|
| `text2sql_eval.jsonl` | 20 | employees | the reference set |
| `text2sql_eval_paraphrase.jsonl` | 20 | employees | **wording only**: same 20 golds, reworded |
| `text2sql_eval_bookstore.jsonl` | 20 | bookstore | **schema only**: same intents, all-new names |
| `text2sql_eval_join.jsonl` | 12 | employees | **construct**: joins on a TEXT key + a self-join |
| `text2sql_eval_join_bookstore.jsonl` | 11 | bookstore | join key becomes an **integer FK** |
| `text2sql_eval_blind_v1_retired.jsonl` | 24 | employees | **spent** blind set #1, now a regression set |
| `text2sql_eval_blind_v2_retired.jsonl` | 30 | employees | **spent** blind set #2, now a regression set |
| `text2sql_eval_blind_v3.jsonl` | 30 | employees | **held back**: independently authored, scored once |

Holding everything else fixed is what makes a score drop *diagnostic* rather than merely bad.

The first seven are **development sets**: their failures were read and acted upon, which is how
the training data improved, and which is also why they can no longer be called unbiased. The
eighth is held back, excluded from `make eval-all`, and `tests/test_eval_blind.py` fails if
anyone quietly adds it to the regression loop.

### The leakage contract

A training candidate is dropped if its normalised question **or** normalised SQL collides
with any eval example, and the blocklist is built from **every** file in `data/eval`. The SQL
check reuses `src.metrics.normalize_sql`, the exact function used for scoring, so a training
target can never equal a graded answer. Each build also *reports* the highest word overlap
between any training and eval question, so "not even close" is measured rather than asserted.

A consequence worth stating plainly: a pattern whose SQL *is* an eval gold (e.g.
`SELECT name FROM departments`) gets **zero** training examples and must be reached by
generalisation. `tests/test_build_dataset.py` asserts all of this against the generated data.

---

## Results

Base model vs. the frozen adapter, both metrics, all seven development sets:

| Eval set | n | base EM | base exec | **LoRA EM** | **LoRA exec** |
|---|:--:|:--:|:--:|:--:|:--:|
| in-template | 20 | 40% (8) | 65% (13) | **100% (20)** | **100% (20)** |
| out-of-template | 20 | 30% (6) | 55% (11) | **100% (20)** | **100% (20)** |
| cross-schema (bookstore) | 20 | 55% (11) | 70% (14) | **100% (20)** | **100% (20)** |
| JOIN (employees, text key) | 12 | 0% (0) | 8% (1) | **100% (12)** | **100% (12)** |
| JOIN (bookstore, integer FK) | 11 | 0% (0) | 64% (7) | 91% (10) | 91% (10) |
| retired blind v1 | 24 | 21% (5) | 29% (7) | **75% (18)** | **88% (21)** |
| retired blind v2 | 30 | 10% (3) | 10% (3) | **30% (9)** | **40% (12)** |
| **development total** | **137** | **24% (33)** | **41% (56)** | **79% (108)** | **84% (115)** |

And the held-back set, scored exactly once against the frozen model:

| Blind set v3 | n | base EM | base exec | **LoRA EM** | **LoRA exec** |
|---|:--:|:--:|:--:|:--:|:--:|
| easy | 8 | n/a | 37% (3) | n/a | **87% (7)** |
| medium | 14 | n/a | 7% (1) | n/a | **64% (9)** |
| hard | 8 | n/a | 0% (0) | n/a | **12% (1)** |
| **total** | **30** | **3% (1)** | **13% (4)** | **40% (12)** | **57% (17)** |

The adapter adds **4.4M** trainable parameters (0.88%) on top of the frozen 0.49B base and
trains in ~20–40 minutes on a laptop.

**One recovered regression, and the two failed attempts that came first.** The cross-schema FK
join set fell from 100% to 82% when the new construct families landed. A training-data fix took
it to 91%; two others made it worse and were reverted (`…-constructs-rebalanced`,
`…-joinorder` in `results/baseline.md`). What finally closed it was not training data at all:
the [execute-and-repair loop](#execute-and-repair-a-win-on-one-set-and-a-null-result-on-the-one-that-matters)
returns it to **100%** at inference time. The table above is the model on its own; with
`--repair 2` the development total is 97%.

---

## Three blind sets, and what their disagreement measures

This is the part of the project I would most want a reviewer to read.

A held-back set is only unbiased while nobody acts on it. Read its failures, act on them, and
it has become development signal — whatever the file is still called. Most projects quietly
keep quoting the old number anyway. This one has been through that cycle three times, and the
bookkeeping is the point:

| | author | scored | fate |
|---|---|:--:|---|
| **v1** (24 q) | me, the data curator | 29% → **75%** | spent: its failures motivated seven construct families → retired into `eval-all` |
| **v2** (30 q) | independent agent | 10% → **40%** | spent: its failures exposed the join-grouping starvation and the vocabulary gap → retired into `eval-all` |
| **v3** (30 q) | a second independent agent | 13% → **57%** | **current.** Scored once against the frozen model, failures deliberately unread |

**Same task, same protocol, three numbers: 75%, 40%, 57%.** The model did not swing that
wildly — on v1 it *improved* from 75% to 88% while v2 said 40%. What changed was who wrote the
exam. That spread is the most useful thing measured here, and it is why every number in this
README is quoted with its provenance attached.

Some of the spread is diagnosable. v1 was written by me and scored highest, which is exactly
the bias you would predict. v2 and v3 were both independent, yet differ by 17 points, and a
large part of that is **vocabulary**: v2's author said "team" where the schema says
`department`, so the model invented a `teams` table; v3's author wrote "employee department
label", which maps straight onto the column. Neither is wrong — real users write both — and the
gap between them is a real, measurable fragility that a single blind set would have hidden
entirely.

### How independence is enforced

Each replacement author is an agent given read access to **exactly two files** — the schema and
the seed rows — and denied the training generator, every existing eval set, the results
directory, the READMEs and all model output. It is told to write what a data analyst would
actually ask, with a fixed difficulty spread, and is **not** told which SQL constructs this
project teaches.

Review touches well-posedness only, and always before any model is run:

- **Decorative `ORDER BY` is removed.** The scorer treats an ordered gold as order-sensitive,
  so an `ORDER BY` the question never asked for grades a correct answer wrong.
- **Under-specified questions are made explicit** about which columns to return.
- **Golds that are training examples are replaced.** Three of v3's first draft were verbatim
  training targets — over a two-table schema the supply of canonical single-condition queries
  is small and the generator already covers most of it. A held-out question whose exact answer
  was a training example measures memorisation, so those three were sent back.

The bar v3 clears that v2 did not: **adding it to `data/eval` changed zero training examples.**
The training split is byte-identical before and after, so its score describes exactly the
frozen adapter, with no retraining and no moving target. Four of its golds do coincide with
development-set answers; that is duplication, not contamination — the model was never trained
on those either — and `tests/test_eval_blind.py` bounds it at a fifth of the set.

**Stated as a limitation rather than a win:** a blind set written by the model's own author
systematically overestimates it, and two independent authors can still disagree by 17 points.
Any single held-out number — including the 57% at the top of this README — is one sample from
that distribution.

---

## Is the fine-tune worth it? (vs. a 3x larger model)

Every number above compares the fine-tune to its own base model, which is the wrong baseline
for the decision a reader actually faces: **fine-tune a small model, or just use a bigger
one?** So `Qwen2.5-1.5B-Instruct` (3x the parameters, no fine-tuning, same prompt, same
greedy decoding) was scored on every set.

| Eval set | n | 0.5B base | **1.5B zero-shot** | **0.5B + LoRA** |
|---|:--:|:--:|:--:|:--:|
| in-template | 20 | 65% | 95% | **100%** |
| out-of-template | 20 | 55% | 85% | **100%** |
| cross-schema | 20 | 70% | 100% | **100%** |
| JOIN (employees, text key) | 12 | 8% | 67% | **100%** |
| JOIN (bookstore, integer FK) | 11 | 64% | **91%** | **91%** |
| retired blind v1 | 24 | 29% | 75% | **88%** |
| retired blind v2 | 30 | 10% | **43%** | 40% |
| *development total* | *137* | *41%* | *77%* | ***84%*** |
| **blind v3 (easy)** | **8** | **37%** | **75%** | **87%** |
| **blind v3 (medium)** | **14** | **7%** | **35%** | **64%** |
| **blind v3 (hard)** | **8** | **0%** | **12%** | **12%** |
| **blind v3 (total)** | **30** | **13%** | **40%** | **57%** |

<sub>Execution accuracy. Exact-match across the development sets: 24% / 44% / 79%.</sub>

**On the development sets the fine-tune wins** (84% vs 77%) at a third of the size, and it
matches the house SQL style far more often (79% vs 44% exact-match). That is the standard
argument for task-specific fine-tuning, and it holds.

**On the independently written blind set it wins by more: 57% vs 40%**, a 5-question margin,
driven almost entirely by the medium tier (64% vs 35%). That is the result the project exists
to support, and it is worth noting how nearly it went the other way: on the *previous* blind
set the same two models finished 43% to 40% in the 1.5B's favour. Same models, same protocol,
different author. Where they differ:

- For questions inside a known, narrow distribution, **4.4M trained parameters buy more than 1B
  extra frozen ones**. The 0.5B fine-tune runs on a laptop CPU.
- The fine-tune's advantage is concentrated in **grouping, joins and having** — the taxonomy it
  was taught. The 1.5B over-engineers those, building needless subqueries and joins.
- For genuinely compositional SQL, **neither approach works at this scale**: 1/8 each.
  Reaching that tier needs a different intervention (composition in the training data, a
  larger base, or an agentic loop that checks its own SQL), not more of what is here.

**The honest read:** if the query distribution is known and narrow, the 0.5B fine-tune is the
better deal. If it is open-ended, the bigger model is the safer default, and the fine-tune's
development-set advantage partly reflects that those sets were used to develop it. Neither
statement would have been visible without an independently written blind set.

---

## Deploying it: quantization and a small API

### Execute-and-repair: a win on one set, and a null result on the one that matters

The API already *runs* the SQL it generates, so when a query is invalid SQLite says exactly
what is wrong with it. Feeding that back is free information at inference time and needs no
retraining: `--repair 2` re-asks once, with the error translated into an instruction.

The headroom was measured before the mechanism was built. Of the adapter's 22 failures across
all seven eval sets, **15 raise a SQLite error** and 7 run fine but return the wrong rows —
so the ceiling is "some of 15". Without the gold there is no way to know that a query which
executed cleanly answered the wrong question, and the loop never sees the gold. It also cannot
lower a score: it fires only for a query that already failed to execute, and such a query is
already graded wrong. That is a property of the design, and `tests/test_repair.py` pins it.

| | JOIN (bookstore, FK) | development total | **blind v3** |
|---|:--:|:--:|:--:|
| single generation | 91% | 84% | **57%** |
| `--repair 2` | **100%** | **85%** | **57%** |

**It fully recovered the cross-schema FK join regression** — 91% → 100%, at inference time,
with no retraining — and bought **exactly nothing** on either held-back set. On blind v2 it
made 3 queries runnable and turned none of them into correct answers; on blind v3 it never
fired at all, because the frozen model produced runnable SQL for all 30 questions. Every one
of its 13 failures there is valid SQL answering the wrong question — precisely the category the
loop cannot see.

That gap is the interesting part. The loop fixes *lookup* errors, where the right identifier
exists and is nearby: told `no such column: publishers.publisher_id`, the model re-reads the
schema and joins on `publishers.id`. It does nothing for *vocabulary* mismatches, and blind v2
is full of them — its author says "team" where the schema says `department`, so the model
invents a `teams` table and, told there is no such table, invents it again. In 10 of 13 retries
the second attempt reproduced the identical error. Nor does it help the compositional failures,
where the query errors because the model is attempting something it cannot do at all.

Two smaller findings worth recording, both from probing the model directly rather than guessing:

- **Never show the model its own failed SQL.** The first version put the failed query and the
  raw SQLite message in the prompt, and every retry came back *byte-identical* to the first
  attempt. Under greedy decoding the previous answer is a strong prior to reproduce it.
- **Make the instruction corrective, not prohibitive.** "Do NOT use the column
  `publishers.publisher_id`" made the model avoid the whole table and invent `cities.city`
  instead. "That column does not exist; use the correct one from the schema" produced the gold
  query exactly.

**The honest read:** worth shipping (it is cheap, it cannot hurt, and it recovered a real
regression), but it is not a route to the hard tier. It is also more useful to a *weaker* model
— on the base model it lifted blind v2 from 10% to 13%, because more of the base model's
failures are the kind a schema hint can fix.

<details>
<summary><b>Quantization: measured, and not recommended at this size</b></summary>

Quantization is implemented (`--quantize`, dynamic int8 on the linear layers, CPU-only per
PyTorch) and, because "quantized successfully" is not a result, it was scored rather than
merely demonstrated:

| | size | in-template | blind | latency (blind) |
|---|:--:|:--:|:--:|:--:|
| fp32 (CPU) | 1976 MB | **95%** | **75%** | **20.2s** |
| int8 dynamic (CPU) | **1041 MB** | 75% | 21% | 25.8s |

**It loses on every axis that matters.** Memory halves, which is real, but in-template accuracy
drops 20 points, the blind set collapses, and it is *slower*: dynamic quantization
re-quantizes activations per forward pass and Apple Silicon has no optimised int8 path for
these kernels.

That is unsurprising for a 0.5B model: there is little redundancy left to discard, so
aggressive post-training quantization removes signal rather than slack. It pays off at 7B+, or
with a calibration set and static quantization, neither of which this project has.
**The feature ships; the recommendation is not to use it**, and the numbers explaining why ship
with it.

<sub>fp32 is measured on CPU too, because comparing an int8 CPU run against an fp32 GPU run would
confound quantization with the device. Quantizing a `PeftModel` directly would leave the LoRA
layers in float and quantize *around* them, so the adapter is merged into the base first.
These numbers are from the previous adapter and have not been re-run this round.</sub>

</details>

`src/serve.py` puts the model behind an HTTP endpoint. Because the schema has a real seeded
database behind it, the API **runs** the SQL it generates and returns rows (question → SQL →
answer, end to end):

```console
$ make serve
Serving Qwen/Qwen2.5-0.5B-Instruct + lora-qwen2.5-0.5b-joingroup on http://127.0.0.1:8000

$ curl -s localhost:8000/sql -d '{"question": "How many employees are in the Sales department?"}'
{
  "question": "How many employees are in the Sales department?",
  "sql": "SELECT COUNT(*) FROM employees WHERE department = 'Sales'",
  "schema": "employees",
  "columns": ["COUNT(*)"],
  "rows": [[4]],
  "row_count": 1
}
```

Pass `"schema": "bookstore"` to query the other schema, `"execute": false` for SQL only, or
`"repair": 2` to enable the retry loop for that request (`--repair 2` sets the default).
Two deliberate choices:

- **No web framework.** `http.server` from the standard library. CI installs neither torch nor
  transformers, and a text-to-SQL demo does not need FastAPI to prove anything.
- **Executing model output demands a guard.** Generated SQL is refused unless it is a single
  read-only `SELECT`, so a hallucinated `DROP TABLE` never reaches the database, and a broken
  query returns a 422 with the error rather than a stack trace.

The whole request path (routing, validation, status codes, the safety filter, execution) is
covered by `tests/test_serve.py` **without torch**: `src/serve.py` defers every heavy import
and expresses its logic against a plain `generate(question, schema) -> sql` callable, so a stub
generator exercises all of it in milliseconds on a CI box with no model.

---

## Quickstart

**Requirements:** Python 3.9+, macOS or Linux. Apple Silicon GPU (MPS) is used automatically;
no NVIDIA card is needed.

```bash
make setup       # venv + torch / transformers / datasets / peft / accelerate
make smoke       # fast end-to-end pipeline check on a tiny model (no big download)
make baseline    # reproduce the "before" number (downloads Qwen2.5-0.5B-Instruct once)
make data        # (re)generate the de-leaked NL->SQL training set into data/train/
make train       # LoRA fine-tune -> adapters/
make eval-all    # score the adapter on the seven development sets (regression check)
make serve       # serve over HTTP on :8000 (POST /sql)
make test        # 197 unit tests, no model download
```

**Reproducing the headline numbers takes two commands.** `adapters/` is gitignored (trained
weights do not belong in git), so a fresh clone has no adapter and `make eval-ft` will say so.
`make data && make train` builds it; the training data is committed and the generator is
seeded, so the *input* is byte-reproducible.

On a memory-constrained machine, use a smaller micro-batch with matching accumulation. The
**effective** batch size (8) and therefore the optimizer schedule are unchanged:

```bash
make train TRAIN_ARGS="--batch-size 4 --grad-accum 2"
```

<sub>Worth knowing: this round trained faster on **CPU** than on MPS. With the larger training
set, MPS drove the machine into swap and throughput collapsed to a few optimizer steps per
minute; `--device cpu` finished the same run in 17–37 minutes depending on system memory
pressure. Measure before assuming the GPU path is faster.</sub>

Individual targets: `eval-ft` (in-template), `eval-ood` (reworded), `eval-schema` (bookstore),
`eval-join` (both JOIN sets), `make compare` (3x larger model on every set), `make repair`
(every dev set with the error-feedback retry on), `make quantized`
(what int8 costs).

There is also `make eval-blind`, which scores the **held-back** set. It is intentionally not
part of `eval-all`: run it once against a finished model, record the number, and leave its
failures alone. Running it repeatedly while iterating on training data is exactly how a blind
set stops being blind.

```bash
make eval-all ADAPTER=adapters/lora-qwen2.5-0.5b-aug
make compare COMPARE_MODEL=Qwen/Qwen2.5-3B-Instruct
make serve SERVE_ARGS="--quantize --port 9000"
```

---

## Experiment log (how it got here)

Each round has the same shape: build a set that isolates one variable, find a specific
weakness, change **only the training data**, retrain with identical hyper-parameters, and
re-score every set.

<details>
<summary><b>Round 1: a baseline, then a first LoRA</b> (65% → 100% in-template, and why that means little)</summary>

The base model scored **65%** execution accuracy in-template. A LoRA fine-tune on a synthetic,
de-leaked training set took it to **100%**. Impressive, and almost meaningless on its own: the
eval was *in-distribution* with the training templates, so it mostly proved pattern-fit plus
the ability to copy literals out of the question.

</details>

<details>
<summary><b>Round 2: does it survive rewording?</b> (100% → 75%)</summary>

`text2sql_eval_paraphrase.jsonl` keeps the **same 20 golds** and rewrites every question in
unfamiliar, indirect language. Same schema, same DB, so only wording changes.

| Model | in-template | out-of-template |
|---|:--:|:--:|
| base | 65% (13/20) | 55% (11/20) |
| + LoRA | 100% (20/20) | **75% (15/20)** |

LoRA had learned the template patterns strongly but only partly generalised. It still beat the
base on the same set (75% vs 55%), so it learned intent rather than surface strings. The
misses were informative: "our headcount" and "how big is the Sales team" produced
`SUM(salary)`; the "lowest-paid employee" came back as `MIN(salary)` instead of a name.

</details>

<details>
<summary><b>Round 3: does it survive a new schema?</b> (100%, no loss)</summary>

`text2sql_eval_bookstore.jsonl` re-asks the same 20 intents against a completely different
schema (`publishers`, `books`) mirroring the original construct-for-construct with all-new
names.

| Model | employees | bookstore (unseen) |
|---|:--:|:--:|
| base | 65% | 70% |
| + LoRA | 100% | **100%** |

The fine-tune held **100%** with **zero** leakage of employees-schema names. So the
brittleness was specifically about **phrasing**, not **schema**: rewording cost 25 points,
swapping the entire schema cost nothing.

</details>

<details>
<summary><b>Round 4: closing the phrasing gap</b> (75% → 90%)</summary>

Each SQL pattern previously had exactly one wording, precisely what a model overfits to. Now
every pattern ships interchangeable phrasings varying register, synonyms and sentence shape.
Two curation rules came out of it:

- **Balance the pattern mix.** Parameter pools differ wildly in size, so naive expansion made
  some patterns 14x more frequent than others, and the model answered rare patterns with a
  frequent pattern's shape. Capping each pattern at 24 fixed it.
- **Avoid ambiguous wordings.** Describing `SELECT department FROM employees` as "the
  departments of all employees" taught the model that "departments" implies a `department`
  column; it then answered "list all department names" with `SELECT department FROM
  departments`, a column that does not exist.

**+15 points on the weakness, no regression elsewhere.** The two survivors were the same
mistake: "total headcount" and "how big is the Sales team" both produced `SUM(salary)`; the
model read words of magnitude as a request to add up money.

</details>

<details>
<summary><b>Round 5: JOINs, and a capability the fine-tune had destroyed</b> (0% → 100%)</summary>

No set so far contained a `JOIN`, so nothing taught one, and nothing measured whether the
fine-tune could still write one. Scoring the *existing* adapter on two new join sets was the
uncomfortable part:

| Model | JOIN / employees | JOIN / bookstore FK |
|---|:--:|:--:|
| base | 8% (1/12) | **64% (7/11)** |
| + LoRA (round 4) | 0% (0/12) | **9% (1/11)** |

**The fine-tune had destroyed a capability the base model already had.** Three epochs on a
corpus of exclusively single-table queries did not merely fail to teach joins; it taught the
model that queries *are* single-table. That is catastrophic forgetting, and it was invisible
until a set existed to measure it.

The fix was data-only: six join families plus a contrastive magnitude lesson. Joins went to
**100%** on the taught schema and **100%** on the bookstore schema, whose integer-FK join
condition appears *nowhere* in training: the model learned to read the relationship out of the
prompt rather than memorise a condition.

Two further curation rules, both from failures this round's *first* attempt produced:

- **Balance literal shapes, not just pattern counts.** The join `HAVING` family drew only
  single-digit thresholds, and the model answered "more than 10 employees" with
  `HAVING COUNT(*) > 1`; it had learned the digit count, not the number.
- **Do not let a pattern starve.** Adding 133 join examples diluted the smallest patterns, and
  "who is our lowest-paid employee" regressed to `MIN(salary)`.

</details>

<details>
<summary><b>Round 6: teaching what the blind set proved was missing</b></summary>

The first blind measurement said **75%**, and its six failures were diagnostic: four needed a
construct in **no** training template: a year pulled from an ISO date, a bare
`SELECT DISTINCT`, an `IS NULL` test, de-duplicating a join result. The model was not getting
them wrong so much as it had never seen them.

Acting on that spends the set, so the set was [retired and
replaced](#three-blind-sets-and-what-their-disagreement-measures) first. Then seven construct families were added:
`select_distinct`, `null_check`, `date_year`, `like_match`, `between`, `not_equal`, and a
`DISTINCT` self-join. Two of them are deliberately contrastive: `SELECT DISTINCT col` is taught
beside `COUNT(DISTINCT col)` over a shared parameter pool, so "which ones" versus "how many" is
the only thing separating them. The date family teaches `strftime('%Y', hire_date) = '2021'`
specifically because the natural thing to write, `YEAR(hire_date)`, does not exist in SQLite.

The last in-taxonomy dev failure was fixed in the same round. "Show each department and the
number of employees in it" had been answering `SELECT department FROM employees`, because the
leakage filter removes `SELECT department, COUNT(*) FROM employees GROUP BY department` (it
*is* an eval gold), leaving no example of "grouping column + `COUNT(*)` on employees" at all,
so a lexically adjacent projection won the tie. Two families restore the shape without ever
emitting the graded answer: one adds a `WHERE`, the other groups by `manager_id`.

| Eval set | n | base | round 5 | **round 6** |
|---|:--:|:--:|:--:|:--:|
| in-template | 20 | 65% | 95% | **100%** |
| out-of-template | 20 | 55% | 95% | **100%** |
| cross-schema | 20 | 70% | 100% | **100%** |
| JOIN (employees, text key) | 12 | 8% | 100% | **100%** |
| JOIN (bookstore, integer FK) | 11 | 64% | 100% | **82%** |
| retired blind v1 | 24 | 29% | 75% | **88%** |

**A rebalancing attempt that failed, kept in the record.** The FK-join regression looked like
it was caused by the new single-table group-count family outweighing its join counterpart, so a
second build halved the former and added a filtered join group-count. It made things *worse* on
four of six sets, including the one it targeted (82% → 73%), and was reverted. Both runs are in
`results/baseline.md`. Choosing between two candidate adapters on development sets is exactly
what those sets are for, and exactly why the blind set was scored only after that choice was
final.

</details>

<details>
<summary><b>Round 7 — un-starving the join grouping family</b></summary>

Round 6 left one regression: cross-schema FK joins at 82%. The failing items shared a shape,
and #9 on the same set — which also groups on the joined column — *passed*. So it was not
"grouping" that broke; it was specifically `SELECT <grouped column>, <aggregate>`.

Counting that shape in the training data made the cause obvious: **6 examples with a join
against 42 without**, a 7:1 skew. The reason is structural and worth remembering — the two most
representative targets of the joined form, `COUNT(*)` and `AVG(salary)` per location, are *both
eval golds*, so the leakage filter deletes them. What survived was `SUM` and `MAX` only, while
six `HAVING` targets shared the same 24-example cap. The category read as a healthy 24 the whole
time. **Balance has to be checked per SQL shape, not per category label** — this is the round-5
"do not let a pattern starve" rule recurring in a place the existing checks could not see.

The fix has two parts: `HAVING` moves to its own category so it stops consuming the budget, and
filtered variants (`… WHERE employees.salary > n GROUP BY departments.location`) replace the
deleted golds with targets the leakage filter cannot reach. The starved shape went from 6 to 24
examples, and the single-table side was left untouched.

| Eval set | n | round 5 | round 6 | **round 7** |
|---|:--:|:--:|:--:|:--:|
| in-template | 20 | 95% | 100% | **100%** |
| out-of-template | 20 | 95% | 100% | **100%** |
| cross-schema | 20 | 100% | 100% | **100%** |
| JOIN (employees, text key) | 12 | 100% | 100% | **100%** |
| JOIN (bookstore, integer FK) | 11 | 100% | 82% | **91%** |
| retired blind v1 | 24 | 75% | 88% | **88%** |
| **blind v2** (scored once) | 30 | — | 37% | **40%** |

**A second failed fix, also kept in the record.** The one remaining failure is a pure
join-condition error: the model writes
`FROM publishers JOIN books ON publishers.publisher_id = books.publisher_id`, putting the
one-side table first and then repeating the many-side's column on both sides of the `=`.
Everything else about that query is right. Since every training join names `employees` first,
the obvious diagnosis was that the `ON` clause had been learned *positionally* — so the reverse
order was taught explicitly, in its own category, with the leakage blocklist widened to cover
both spellings of every join gold.

**It was a disaster: FK joins collapsed to 18% (2/11).** The model started mixing undeclared
aliases into the `SELECT` list, joining on `publishers.name`, and inverting the order even more
often. The lesson is the opposite of the hypothesis: at this scale the model needs a *consistent*
convention more than it needs to know that two spellings are equivalent. Ambiguity in the
training signal cost more than positional memorisation did. Reverted.

The blocklist hardening was kept even though the family that motivated it was removed. It is a
no-op today (zero candidates are caught only by it), but `normalize_sql` compares strings, so
`FROM a JOIN b ON a.x = b.y` and its mirror image would otherwise look like two different
targets, and a reversed graded answer could enter training as a "new" one. That hole is one edit
away from mattering; `tests/test_build_dataset.py` now pins it shut.

</details>

<details open>
<summary><b>Round 8 — the execute-and-repair loop, and closing the project</b> (final)</summary>

Two things happened in the last round, and only one of them was a model change.

**The repair loop.** The API already runs the SQL it generates, so an invalid query comes back
with SQLite's own diagnosis. Feeding that back is free at inference time. It recovered the last
cross-schema FK join (91% → **100%**) and bought **nothing** on either held-back set. Two design
findings came out of probing rather than guessing: never show the model its own failed SQL
(under greedy decoding it simply reproduces it — the first version's retries were
byte-identical), and phrase the feedback as a correction rather than a prohibition ("that column
does not exist, use the correct one" fixed a query that "do NOT use that column" broke further,
by making the model avoid the whole table).

**Retiring v2 and commissioning v3.** By this point v2's failures had been read twice — once for
the tier breakdown, once to analyse what the repair loop did — and both remaining work items
were derived from them. That spends a blind set, whatever the file is still called, so v2 was
retired into `eval-all` and a second independent author wrote v3 against the frozen model.

v3 clears a bar v2 did not: **adding it changed zero training examples**, so its score describes
exactly the shipped adapter with no retraining. Getting there took one revision round — three of
its first-draft golds were verbatim training targets, because the supply of canonical
single-condition queries over a two-table schema is small and the generator already covers most
of it.

| | base 0.5B | 1.5B zero-shot | **0.5B + LoRA** |
|---|:--:|:--:|:--:|
| development sets (137 q) | 41% | 77% | **84%** |
| **blind v3** (30 q, scored once) | 13% | 40% | **57%** |

The fine-tune beats the 3x larger model on both — and on the *previous* blind set the same two
models finished 40% to 43% the other way. Same models, same protocol, different author. That is
the note the project ends on.

</details>

### Where it stops, and why that is the end

The model is frozen. These are the limits it was frozen with — stated as scope, not as a
backlog, because closing any of them would cost another blind set:

- **Compositional SQL is out of reach.** 1/8 on the hard tier, and the 1.5B also manages 1/8.
  Correlated subqueries, `NOT EXISTS`, derived tables and per-group extremes are simply not
  something this approach delivers at 0.5B. Closing it means compositional training data or a
  bigger base model — a different project, not another round of this one.
- **Vocabulary is brittle.** Blind v2's author said "team" where the schema says `department`
  and the model invented a `teams` table, twice, even when told it did not exist. Blind v3's
  author said "employee department label" and the model was fine. That fragility is worth
  roughly 17 points between two independent authors, and it is the largest single lever left.
- **One cross-schema FK join fails on a single generation** (10/11). Two training-data
  hypotheses were tried and refuted; the execute-and-repair loop closes it at inference time.
- **The blind sets' failures are deliberately not being fixed.** Reading them as a to-do list is
  exactly what spent v1 and v2. They are recorded as findings. Anyone continuing this work
  should commission v4 *first*.

---

## Honest caveats

- **The wins are leakage-free.** No eval question or gold SQL appears in training, enforced in
  `src/build_dataset.py` and asserted in the test suite.
- **84% is development signal; 57% is the estimate.** Seven data-curation decisions were
  diagnosed by reading dev-set failures, so those sets are no longer unbiased. Nothing was ever
  tuned *to the answers* (the leakage filter guarantees that), but the honest headline is the
  held-back number.
- **Blind sets are single-use, and two have been used.** v1 and v2 were each scored once, then
  spent when their findings were acted upon; both are now labelled and scored as development
  sets. Any *further* claim of an unbiased estimate requires a new set.
- **The three blind sets are not comparable with each other.** Different authors, different
  difficulty distributions. 75% → 40% → 57% is not a trajectory; on v1 the model *improved*
  from 75% to 88% during the period v2 read 40%. Quote them separately, with their provenance,
  or not at all.
- **57% is one sample.** Two independent authors, same brief, same schema, disagreed by 17
  points. A single held-out number carries more uncertainty than its confidence interval
  suggests, and this repo can put a figure on that only because it built three.
- **Two rounds' worth of fixes were rejected on the evidence.** Three candidate adapters were
  built to recover one regression; two made things worse and were reverted, and all three are in
  `results/baseline.md`. Choosing between candidates on development sets is what those sets are
  for, and it is exactly why the blind set is scored only after the choice is final.
- **The in-template set remains in-distribution** with the synthetic training patterns (same SQL
  shapes, unseen literals). The other sets exist because that number alone would overstate the
  model.
- **Everything here is one small model on two small schemas.** 11–30 questions per set means a
  single example is worth 3–9 points, so small differences are noise. The point of the repo is
  the *method* (measure first, isolate one variable per set, keep one set honest), not the
  absolute scores.
- **The adapter is not in the repo.** `adapters/` is gitignored, so a fresh clone reproduces the
  numbers with `make data && make train`. The training data *is* committed and the generator is
  seeded, so the input is byte-reproducible; trained weights are not bit-identical across
  devices, and small eval wobble between hardware is expected.

---

## Repo layout

```
.
├── src/
│   ├── eval_baseline.py   # run a model on an eval set, save the before/after numbers
│   ├── build_dataset.py   # synthesise the de-leaked NL->SQL training set
│   ├── train_lora.py      # LoRA fine-tune the base model on data/train/
│   ├── serve.py           # HTTP API: question -> SQL -> rows (stdlib only)
│   ├── repair.py          # execute-and-repair: re-ask with SQLite's error (stdlib only)
│   ├── data_utils.py      # both schemas + shared prompt format (keeps train == eval)
│   ├── db.py              # seeded SQLite DBs + execution-accuracy scoring
│   └── metrics.py         # SQL normalisation + exact-match scoring
├── data/
│   ├── eval/              # 6 development sets + 1 held-back blind set, and a data card
│   └── train/             # generated train/val split + a data card
├── tests/                 # stdlib-only unit tests (no torch), run in CI
├── results/
│   ├── baseline.md        # human-readable leaderboard
│   └── *.json             # full per-example predictions (committed)
├── requirements.in        # local (CPU/MPS) top-level deps
├── requirements.txt       # pinned lockfile (pip freeze)
├── requirements-gpu.txt   # CUDA-only extras (bitsandbytes)
├── requirements-dev.txt   # lint + test only, used by CI
├── LICENSE                # MIT
└── Makefile               # setup / smoke / baseline / data / train / eval-* / serve / test
```

### Method in one screen

- **Prompt format:** schema + question via the model's own chat template, *identical* for
  training and evaluation to avoid prompt-format drift, a classic silent-failure bug.
  → `src/data_utils.py`
- **Training loop:** an explicit PyTorch loop rather than `Trainer`, so every mechanism is
  visible: LoRA on attention + MLP projections, **prompt masking** (loss on SQL tokens only),
  gradient accumulation, warmup + linear decay, train/val loss watched together because a tiny
  dataset overfits fast. → `src/train_lora.py`
- **Generation:** greedy and deterministic, so runs are reproducible and before/after
  comparisons are fair. → `src/eval_baseline.py`
- **Output:** a machine-readable `results/*.json` per run (every prediction, both metrics) plus
  a human-readable `results/baseline.md` leaderboard.

---

## What was built, in order

Every item was measured before and after; the rounds that made things worse are in the
[experiment log](#experiment-log-how-it-got-here) alongside the ones that worked.

- ✅ De-leaked NL→SQL **training set**, kept strictly separate from eval.
- ✅ **LoRA fine-tune** + **execution-accuracy** scoring on a seeded SQLite DB.
- ✅ **Out-of-template phrasing** eval, quantifying a 100% → 75% generalisation gap, then
  closing it with paraphrase augmentation and a balanced pattern mix.
- ✅ **Cross-schema** eval: the fine-tune holds 100% on a schema it never trained on.
- ✅ **Multi-table `JOIN`** evals, which exposed catastrophic forgetting (base 64% → 9%), then
  six join families taking it to 100% on both, including a join key absent from training.
- ✅ **Benchmarked against a 3x larger model** and **served behind an API**; int8 quantization
  implemented, measured, and recommended against at this size.
- ✅ **Three blind sets**, each scored once — the first written by me, the next two by
  independent authors under enforced isolation. Two were spent and retired into the regression
  loop; the third is the final estimate.
- ✅ **Taught the constructs v1 proved were missing** (`IS NULL`, `SELECT DISTINCT`, `strftime`
  dates, `LIKE`, `BETWEEN`, `!=`) and closed the last in-taxonomy dev failure.
- ✅ **Found and fixed a silently starved pattern**: the join-grouping family had 6 training
  examples against 42 single-table ones, because its two most natural targets are eval golds
  and the leakage filter deletes them. Balance is now asserted per SQL *shape*, not per
  category label.
- ✅ **An execute-and-repair loop** that re-asks with SQLite's own error, recovering the last
  cross-schema FK join at inference time — and demonstrably buying nothing on either held-back
  set.

### If someone picks this up

The model is frozen and the remaining limits are documented
[above](#where-it-stops-and-why-that-is-the-end). Anything further starts the same way:
**commission blind set v4 first**, from a new author under the same isolation, because v2 and
v3 have both now been read. Then, in rough order of expected value: synonym grounding for
schema vocabulary, compositional training data for the hard tier, and a larger base model if
neither is enough.

---

## Appendix

**Compute & environment.** Local: Apple Silicon macOS, CPU/MPS. A cloud **T4** (Colab / Kaggle
free tier) is the fallback for larger runs; `bitsandbytes` / 4-bit QLoRA is **CUDA-only**, so it
lives in `requirements-gpu.txt` and is not installed locally.

**Tech stack.** Python · PyTorch · Hugging Face `transformers`, `datasets`, `peft` (LoRA),
`accelerate`. Experiment tracking: JSON/CSV. CI runs ruff + pytest on the deterministic core
only: no torch, no model download, so it finishes in ~30 seconds.

**Evaluation principles.** Measure a baseline *before* training, or improvement cannot be
proven. Watch eval loss, not just train loss. Keep tokenization and prompt format identical
between training and evaluation. Never claim a result that is not on a held-out set. And when a
held-out set stops being held out, say so.

**Why this project.** An end-to-end demonstration of small-LLM post-training and honest
evaluation (data curation, LoRA fine-tuning, memory troubleshooting, regression diagnosis and a
rigorous before/after comparison) rather than calling a hosted API.

---

## License

[MIT](LICENSE). The code, the eval sets, the generated training data and the committed results
are all free to use, modify and redistribute with attribution.

Two things here are **not** covered by that licence, because they are not mine to license:

- **The base model.** `Qwen2.5-0.5B-Instruct` (and the `Qwen2.5-1.5B-Instruct` used in the
  comparison) are released by Alibaba Cloud under **Apache-2.0**. Nothing here redistributes
  their weights; the scripts download them from Hugging Face at run time.
- **Anything derived from those weights.** The LoRA adapter is a delta on top of a Qwen model,
  so a trained adapter carries the base model's terms with it. `adapters/` is gitignored, so this
  repo ships none; `make data && make train` builds one locally.

In short: the *method*, the data and the measurements are MIT; the *model* stays under
Apache-2.0.
