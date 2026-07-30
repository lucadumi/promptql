# PromptQL - fine-tune a small LLM for natural language to SQL

> Take a small, **open** language model I fully control, teach it to translate plain-English
> questions into SQL, and **prove** it improved with an honest before/after evaluation.

A complete **post-training + evaluation loop** in miniature: pick a small open model, measure
it first, fine-tune it with LoRA on a narrow task, then re-measure on held-out sets and be
honest about what got *worse*, not just what got better. It is deliberately small and cheap
so the whole loop runs on a laptop - the point is the *method*, not the model size.

**Status:** ✅ fine-tuned and evaluated on five development sets **plus one held-back blind
set**. A LoRA adapter training only **0.88%** of the parameters takes
`Qwen2.5-0.5B-Instruct` from **55% → 98%** execution accuracy on the sets used during
development - and, on a set written *after* the model was frozen and scored exactly once,
from **29% → 75%**.

| Eval set | n | what it isolates | base | **+ LoRA** |
|---|:--:|---|:--:|:--:|
| in-template | 20 | the trained SQL patterns, unseen literals | 65% | **95%** |
| out-of-template | 20 | unfamiliar **phrasing**, same golds | 55% | **95%** |
| cross-schema | 20 | an unseen **schema** (bookstore) | 70% | **100%** |
| JOIN (text key) | 12 | multi-table joins + a self-join | 8% | **100%** |
| JOIN (integer FK) | 11 | joins on a key **absent from training** | 64% | **100%** |
| *all development sets* | *83* | | *55%* | *98%* |
| **held-back (blind)** | **24** | **fresh intents, incl. untaught constructs** | **29%** | **75%** |

<sub>Execution accuracy; exact-match is 30% → 95% (dev) and 21% → 71% (blind). Greedy
(deterministic) decoding on Apple Silicon (MPS). Full per-example predictions in `results/`.</sub>

**Read the last row first.** 98% is what the model scores on sets whose failures were read
and acted upon during development; **75% is the honest estimate** of what it does on
questions nobody tuned anything toward. The gap between those two numbers is the single most
useful thing this repo measures - see [the blind result](#the-blind-result-what-it-scores-when-nobody-is-steering).

**Jump to:** [The task](#the-task-text-to-sql) · [How it is measured](#how-it-is-measured) ·
[Results](#results) · [Experiment log](#experiment-log-how-it-got-here) ·
[Honest caveats](#honest-caveats) · [Quickstart](#quickstart) ·
[Method](#how-it-works-method) · [Repo layout](#repo-layout) · [Roadmap](#roadmap)

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

Note that `employees.department` holds a department *name*, not an id - the schema is
denormalised on purpose, and that turns out to matter
(see [round 5](#round-5-joins-and-a-capability-the-fine-tune-had-destroyed)).

---

## How it is measured

### Two metrics, because one of them lies

- **Exact-match** - normalise prediction and gold (strip markdown fences, lowercase, collapse
  whitespace, unify quotes, drop trailing `;`) and compare as strings → `src/metrics.py`.
  Strict, and a *conservative lower bound*: a semantically correct query that differs by one
  keyword is counted wrong.
- **Execution accuracy** - run both queries against a real **seeded SQLite database** and
  compare the returned rows, order-sensitive only when the gold uses `ORDER BY` → `src/db.py`.
  This credits correct-but-differently-written queries.

The gap between them is exactly those queries. On the in-template set the base model answers
**13/20** correctly but phrases 5 of them unlike the gold string - `COUNT(id)` vs `COUNT(*)`,
`SELECT DISTINCT name` vs `SELECT name`, `ORDER BY salary` vs `ORDER BY salary ASC` - so
exact-match says 40% and execution accuracy says 65%. **65% is the honest "before".**

The seed database (`src/db.py`) is deterministic and committed, hand-chosen so every eval
query returns a *discriminating* result: distinct salaries, one department with >10 employees,
budgets/locations/hire-dates that straddle the eval thresholds. A wrong query returns
different rows.

### Six eval sets: five for development, one held back

| File (`data/eval/`) | n | Schema | Changes vs. the in-template set |
|---|:--:|---|---|
| `text2sql_eval.jsonl` | 20 | employees | - (the reference set) |
| `text2sql_eval_paraphrase.jsonl` | 20 | employees | **wording only** - same 20 golds, reworded |
| `text2sql_eval_bookstore.jsonl` | 20 | bookstore | **schema only** - same intents, all-new names |
| `text2sql_eval_join.jsonl` | 12 | employees | **construct** - joins on a TEXT key + a self-join |
| `text2sql_eval_join_bookstore.jsonl` | 11 | bookstore | join key becomes an **integer FK** |
| `text2sql_eval_blind.jsonl` | 24 | employees | **held back** - fresh intents, incl. untaught constructs |

Holding everything else fixed is what makes a score drop *diagnostic* rather than merely bad.

The first five are **development sets**: each one's failures were read and acted upon, which
is precisely how the training data improved. That also means they can no longer be called
unbiased. The sixth is **held back** - written after the shipped model was frozen, scored
once, excluded from `make eval-all`, and never used to steer a curation decision. `make
eval-blind` runs it deliberately, and `tests/test_eval_blind.py` fails if anyone quietly adds
it to the regression loop.

### The leakage contract

A training candidate is dropped if its normalised question **or** normalised SQL collides
with any eval example, and the blocklist is built from **every** file in `data/eval`. The SQL
check reuses `src.metrics.normalize_sql` - the exact function used for scoring - so a training
target can never equal a graded answer. The build also *reports* the highest word-overlap
between any training and eval question, so "not even close" is measured rather than asserted.

A consequence worth stating: a pattern whose SQL *is* an eval gold (e.g.
`SELECT name FROM departments`) has **zero** training examples and must be reached by
generalisation. `tests/test_build_dataset.py` asserts all of this on the generated data.

---

## Results

Per-set numbers for the base model and the current adapter, both metrics:

| Eval set | n | base EM | base exec | **LoRA EM** | **LoRA exec** |
|---|:--:|:--:|:--:|:--:|:--:|
| in-template | 20 | 40% (8) | 65% (13) | **95% (19)** | **95% (19)** |
| out-of-template | 20 | 30% (6) | 55% (11) | **90% (18)** | **95% (19)** |
| cross-schema (bookstore) | 20 | 55% (11) | 70% (14) | **100% (20)** | **100% (20)** |
| JOIN (employees, text key) | 12 | 0% (0) | 8% (1) | **100% (12)** | **100% (12)** |
| JOIN (bookstore, integer FK) | 11 | 0% (0) | 64% (7) | **91% (10)** | **100% (11)** |
| **development total** | **83** | **30% (25)** | **55% (46)** | **95% (79)** | **98% (81)** |
| **held-back (blind)** | **24** | **21% (5)** | **29% (7)** | **71% (17)** | **75% (18)** |

The adapter adds **4.4M** trainable parameters (0.88%) on top of the frozen 0.49B base and
trains in ~17 minutes on a laptop GPU. How it got there is below - the intermediate tables
deliberately show *earlier* adapters, because two of the rounds are about regressions.

---

## Experiment log (how it got here)

Each round follows the same shape: build a set that isolates one variable, discover a
specific weakness, change **only the training data**, retrain with identical
hyper-parameters, and re-score every set.

### Round 1: a baseline, then a first LoRA

The base model scored **65%** execution accuracy in-template. A LoRA fine-tune on a
synthetic, de-leaked training set took it to **100%**. Impressive, and almost meaningless on
its own: the eval was *in-distribution* with the training templates, so it mostly proved
pattern-fit plus the ability to copy literal values out of the question.

### Round 2: does it survive rewording?

`text2sql_eval_paraphrase.jsonl` keeps the **same 20 gold queries** and rewrites every
question in unfamiliar, indirect language. Same schema, same seeded DB, so only the wording
changes.

| Model | in-template (exec) | out-of-template (exec) |
|---|:--:|:--:|
| base | 65% (13/20) | 55% (11/20) |
| + LoRA | 100% (20/20) | **75% (15/20)** |

**100% → 75%.** LoRA had learned the template patterns strongly but only partly generalised
to new phrasings. It still beat the base model on the same set (75% vs 55%), so it learned
intent rather than surface strings. The misses were specific and informative: "our headcount"
and "how big is the Sales team" produced `SUM(salary)` instead of `COUNT(*)`, the
"lowest-paid employee" came back as `MIN(salary)` instead of a name, and "bigger than 10
people" emitted an invalid `WHERE COUNT(*) > 10` instead of `GROUP BY ... HAVING`.

### Round 3: does it survive a new schema?

`text2sql_eval_bookstore.jsonl` re-asks the same 20 intents against a completely different
**bookstore** schema (`publishers`, `books`), built to mirror the original
construct-for-construct with all-new names.

| Model | employees (exec) | bookstore / unseen (exec) |
|---|:--:|:--:|
| base | 65% (13/20) | 70% (14/20) |
| + LoRA | 100% (20/20) | **100% (20/20)** |

The fine-tune held **100%**, and its predictions used the bookstore tables and columns with
**zero** leakage of employees-schema names. So the brittleness was specifically about
**phrasing**, not **schema**: rewording a question cost 25 points, swapping the entire schema
cost nothing.

### Round 4: closing the phrasing gap

Rounds 2 and 3 localised one weakness, so the training data was rewritten to attack it. Each
SQL pattern previously had exactly one question wording - precisely what a model overfits to.
Now every pattern ships a list of interchangeable phrasings varying register, synonyms and
sentence shape, and the generator expands each pattern over all of them (316 examples, up
from 176). Two curation rules came out of it:

- **Balance the pattern mix.** Parameter pools differ wildly in size, so naive expansion made
  some patterns 14x more frequent than others (55 examples vs 4), and the model began
  answering rarer patterns with a frequent pattern's shape. Capping each pattern at 24 fixed
  it.
- **Avoid ambiguous wordings.** Describing `SELECT department FROM employees` as "the
  departments of all employees" taught the model that the word "departments" implies a
  `department` column, and it then answered "List all department names" with
  `SELECT department FROM departments` - a column that does not exist.

| Eval set | before | after |
|---|:--:|:--:|
| in-template | 100% | **100%** |
| out-of-template | 75% | **90%** |
| cross-schema | 100% | **100%** |

**+15 points on the weakness, no regression elsewhere.** Three of the five failures were
fixed. The two survivors were the *same* mistake: "what's our total headcount?" and "how big
is the Sales team?" both produced `SUM(salary)`. The model read words of magnitude as a
request to add up money.

### Round 5: JOINs, and a capability the fine-tune had destroyed

None of the three sets so far contains a `JOIN`, so nothing taught one - and nothing measured
whether the fine-tune could still write one. Two new sets fix that:
`text2sql_eval_join.jsonl` (12 questions on the employees schema, joined on its TEXT key
`employees.department = departments.name`, plus one `manager_id` **self-join**) and
`text2sql_eval_join_bookstore.jsonl` (the same 11 mirrorable intents on the bookstore schema,
related by an **integer foreign key** instead). Every gold filters or groups on a column that
exists only in the *other* table, so a single-table query cannot score by accident.

Scoring the *existing* adapter on them was the uncomfortable part:

| Model | JOIN / employees (exec) | JOIN / bookstore FK (exec) |
|---|:--:|:--:|
| base | 8% (1/12) | **64% (7/11)** |
| + LoRA (round 4) | 0% (0/12) | **9% (1/11)** |

**The fine-tune had destroyed a capability the base model already had.** The base model
writes perfectly reasonable joins on the bookstore schema (`SELECT b.title, p.revenue FROM
books AS b JOIN publishers AS p ON b.publisher_id = p.id`); it fails the employees set almost
entirely for a different reason - it assumes a conventional foreign key, joining
`employees.department` to `departments.id` in 8 of its 12 answers, on a schema whose key is
the department *name*. The fine-tuned model had instead stopped emitting `JOIN` at all,
answering "list each employee's name together with the budget of their department" with
`SELECT name, budget FROM employees`, a column that does not exist on that table. Three
epochs on a corpus of exclusively single-table queries did not merely fail to teach joins; it
taught the model that queries *are* single-table. That is catastrophic forgetting, and it was
invisible until a set existed to measure it.

**The fix, again data-only.** Six join families were added (projection across tables,
filtering and aggregating on a joined column, `GROUP BY`/`HAVING` on it, `ORDER BY`/`LIMIT`
over a filtered join, and the self-join), plus a contrastive magnitude lesson: the count
patterns gained size wordings ("how large is the {dept} team?", "what is the headcount for
{dept}?") and a new per-department `SUM(salary)` pattern ("what is the {dept} department's
total payroll?") sits opposite them, so "total" must be resolved against the noun being
totalled rather than memorised as a word. 501 examples, up from 316.

| Eval set | n | base | round 4 | **round 5** |
|---|:--:|:--:|:--:|:--:|
| in-template | 20 | 65% | 100% | **95%** |
| out-of-template | 20 | 55% | 90% | **95%** |
| cross-schema | 20 | 70% | 100% | **100%** |
| JOIN (employees, text key) | 12 | 8% | 0% | **100%** |
| JOIN (bookstore, integer FK) | 11 | 64% | 9% | **100%** |

<sub>Execution accuracy. Exact-match for round 5: 95 / 90 / 100 / 100 / 91%. One mechanical
difference: this run used a micro-batch of 4 with gradient accumulation 2 rather than a
micro-batch of 8, because the laptop was swapping. The **effective** batch size (8), the
learning-rate schedule and the optimizer step count are identical, so the comparison holds.</sub>

Joins go from **0% to 100%** on the schema they were taught on and - the result that actually
matters - from **9% to 100%** on the bookstore schema, whose integer-FK join condition appears
*nowhere* in training. The model did not memorise a join condition; it learned to read the
relationship out of the schema in the prompt. The magnitude failure is fixed too: "total
headcount" and "how big is the Sales team" now return `COUNT(*)` while "add up all the
salaries" still returns `SUM(salary)`.

**Two more curation rules**, both diagnosed from failures this round's *first* attempt
produced:

- **Balance literal shapes, not just pattern counts.** The join `HAVING` family initially drew
  thresholds from the single-digit end of the pool, and the model answered "more than 10
  employees" with `HAVING COUNT(*) > 1` - it had learned the digit count rather than the
  number. Giving the join families the two-digit tail of the same pool fixed it on both sets.
- **Do not let a pattern starve.** Adding 133 join examples diluted the smallest patterns:
  `extreme_one` fell to 4 examples (its `MIN` target is an eval gold, so the leakage filter
  removes it) and "who is our lowest-paid employee" regressed to `MIN(salary)`. Widening the
  thin patterns with genuinely new targets - most-recent hire, best-funded department,
  `GROUP BY location` on the other table - restored them.

### What is still wrong

One failure remains, and it is the same query on both the in-template and the reworded set:
"show each department and the number of employees in it" returns `SELECT department FROM
employees` instead of `SELECT department, COUNT(*) FROM employees GROUP BY department`.

The diagnosis is measurable rather than speculative. The model produces that exact shape
correctly for `SELECT location, COUNT(*) FROM departments GROUP BY location` (a target it
trained on) **and** for `SELECT genre, COUNT(*) FROM books GROUP BY genre` on a schema it has
never seen - so the shape is learned. It fails only for the `department` / `employees` pair,
which is precisely the one target the leakage filter forbids training to contain, and where a
lexically adjacent projection pattern ("show the department of each employee" →
`SELECT department FROM employees`) wins the tie. Closing it means separating two very
similar sentences, not teaching a new construct.

### The blind result: what it scores when nobody is steering

By this point all five sets above had fed a curation decision, so none of them could still
be called unbiased. `data/eval/text2sql_eval_blind.jsonl` is the correction: **24 fresh
intents**, written after the shipped adapter was trained and frozen, without consulting a
single model prediction. It covers the same taxonomy a user would exercise, plus constructs
that appear in **no** training template - `BETWEEN`, `LIKE`, `IS NULL`, `!=`,
`SELECT DISTINCT`, subqueries and arithmetic between aggregates.

Two properties make the number trustworthy. First, every question and gold was checked
against the generator's *full candidate pool*, not just the written split - so adding the
file changed **zero** training examples, and the score therefore describes exactly the
adapter that was already shipped, with no retraining and no moving target. Second, it was
scored **once**:

| | n | exact-match | exec-accuracy |
|---|:--:|:--:|:--:|
| `Qwen2.5-0.5B-Instruct` (base) | 24 | 21% (5/24) | 29% (7/24) |
| `+ LoRA fine-tune` | 24 | **71% (17/24)** | **75% (18/24)** |

**75%, against 98% on the development sets.** That 23-point gap is the price of having used
those sets to make decisions, and it is the most honest number in this repo. The fine-tune is
still a large, real win - it **more than doubles** the base model (29% → 75%) on questions
nobody tuned anything toward, which is the claim the project actually wants to support.

The six misses are worth reading, because five of them fall outside the syllabus:

| # | Question | What it produced |
|---|---|---|
| 3 | "hired in 2021" | `WHERE YEAR(hire_date) = 2021` - a function SQLite does not have |
| 6 | "the distinct locations" | dropped `DISTINCT` |
| 7 | "employees with no manager" | a self-join instead of `IS NULL` |
| 9 | "department names ordered by budget" | `SELECT department FROM departments` + a stray `LIMIT 1` |
| 22 | "employees who manage someone" | `GROUP BY ... HAVING COUNT(*) > 1` instead of `DISTINCT` |
| 21 | "name next to budget and location" | right rows, **columns in a different order** |

Four of the five real failures (#3, #6, #7, #22) need a construct no training template
contains, which is a coherent and unsurprising story: the model generalises well *within* the
taxonomy it was taught - new phrasings, new schemas, new join keys - and degrades at its
edges. #9 is the one genuine in-taxonomy regression, and it is the same `department`
vs. `departments.name` ambiguity described above. #21 is arguably not a model error at all:
the rows are correct and only the column order differs, which is a limitation of execution
accuracy rather than of the model.

**These failures are deliberately not being fixed.** Reading them as a to-do list is exactly
what would convert this into a sixth development set and leave the project with no unbiased
estimate at all. They are recorded here as findings; the next honest measurement needs a
*new* blind set, not a patched model.

---

## Honest caveats

- **The wins are leakage-free.** No eval question or gold SQL appears in training, enforced in
  `src/build_dataset.py` and asserted in the test suite.
- **The 98% is development signal; the 75% is the estimate.** Four data-curation decisions
  (the balancing cap, the ambiguity fix, the literal-shape balance and the starved-pattern
  fix) were diagnosed by reading failures on the five development sets, so those sets can no
  longer be called unbiased. Nothing was ever tuned *to the answers* - the leakage filter
  guarantees that - but the honest headline is the held-back number:
  [**29% → 75%**](#the-blind-result-what-it-scores-when-nobody-is-steering).
- **The blind set is single-use, and it has now been used.** It was scored once against the
  shipped adapter. Its failures are reported but deliberately not fixed; the moment they are,
  it stops being blind. Any *further* claim of an unbiased estimate requires a **new** set,
  which is why that sits at the top of the roadmap rather than being crossed off.
- **The latest round trades a point.** In-template went 100% → 95% while the reworded set
  gained 5 and the cross-schema join set gained 91. That trade is the honest shape of the
  result, not a rounding error, and it is exactly why every retrain is scored on all five
  development sets rather than on the one it was aimed at.
- **The in-template set remains in-distribution** with the synthetic training patterns (same
  SQL shapes, unseen literal values). The other sets exist because that number alone would
  overstate the model.
- **Everything here is one small model on two small schemas.** 20-24 questions per set means
  a single example is worth 4-5 points, so small differences are noise. The point of the repo
  is the *method* - measure first, isolate one variable per set, and keep one set honest -
  not the absolute scores.

---

## Quickstart

**Requirements:** Python 3.9+, macOS or Linux. On Apple Silicon the Mac GPU (MPS) is used
automatically; no NVIDIA card is needed to *run* the baseline.

```bash
make setup       # create a venv and install torch / transformers / datasets / peft / accelerate
make smoke       # fast end-to-end pipeline check on a tiny model (no big download)
make baseline    # reproduce the 40% baseline (downloads Qwen2.5-0.5B-Instruct once)
make data        # (re)generate the de-leaked NL->SQL training set into data/train/
make train       # LoRA fine-tune on data/train/ -> adapters/ (uses MPS / CPU / CUDA)
make eval-all    # score the adapter on the five development sets (regression check)
make test        # fast unit tests, no model download
```

Individual eval targets: `eval-ft` (in-template), `eval-ood` (reworded), `eval-schema`
(bookstore, base + adapter), `eval-join` (both JOIN sets).

There is also `make eval-blind`, which scores the **held-back** set. It is intentionally not
part of `eval-all`: it is meant to be run once against a finished model, after which its
number should be recorded and its failures left alone. Running it repeatedly while iterating
on the training data is exactly how a blind set quietly stops being blind.

`make data && make train` reproduces the adapter reported above, into
`adapters/lora-qwen2.5-0.5b-join`, which every eval target reads by default. On a
memory-constrained machine use a smaller micro-batch with matching accumulation - the
effective batch size (8), and therefore the optimizer schedule, are unchanged:

```bash
make train TRAIN_ARGS="--batch-size 4 --grad-accum 2"   # ~17 min on an M-series MPS
```

Point any eval target at a different adapter, or try another small model:

```bash
make eval-all ADAPTER=adapters/lora-qwen2.5-0.5b-aug
python -m src.eval_baseline --model Qwen/Qwen2.5-1.5B-Instruct
python -m src.eval_baseline --model HuggingFaceTB/SmolLM2-360M-Instruct --limit 10
```

---

## How it works (method)

- **Prompt format** - schema + question, rendered with the model's own chat template. The
  *exact same* format is used for training and evaluation to avoid prompt-format drift, a
  classic silent-failure bug. → `src/data_utils.py`
- **Training loop** - an explicit PyTorch loop rather than `Trainer`, so every mechanism is
  visible: LoRA on the attention and MLP projections, **prompt masking** (loss on the SQL
  tokens only, `-100` labels elsewhere), gradient accumulation, warmup + linear decay, and
  train/val loss watched together because a tiny dataset overfits fast. → `src/train_lora.py`
- **Training data** - synthesised from templates, several phrasings per SQL target, capped
  per pattern, stratified 10% validation split, fully seeded and reproducible.
  → `src/build_dataset.py`
- **Generation** - greedy / deterministic (`do_sample=False`), so runs are reproducible and
  before/after comparisons are fair. → `src/eval_baseline.py`
- **Scoring** - the two metrics described [above](#two-metrics-because-one-of-them-lies).
  → `src/metrics.py`, `src/db.py`
- **Output** - a machine-readable `results/*.json` per run (every prediction, both metrics)
  plus a human-readable `results/baseline.md` leaderboard.

---

## Repo layout

```
.
├── src/
│   ├── eval_baseline.py   # run a model on an eval set, save the before/after numbers
│   ├── build_dataset.py   # synthesise the de-leaked NL->SQL training set
│   ├── train_lora.py      # LoRA fine-tune the base model on data/train/
│   ├── data_utils.py      # both schemas + shared prompt format (keeps train == eval)
│   ├── db.py              # seeded SQLite DBs + execution-accuracy scoring
│   └── metrics.py         # SQL normalisation + exact-match scoring
├── data/
│   ├── eval/              # 5 development sets + 1 held-back blind set, and a data card
│   └── train/             # generated train/val split + a data card (README.md)
├── tests/                 # stdlib-only unit tests (no torch), run in CI
├── results/
│   ├── baseline.md        # human-readable leaderboard
│   └── *.json             # full per-example predictions (committed)
├── requirements.in        # local (CPU/MPS) top-level deps
├── requirements.txt       # pinned lockfile (pip freeze)
├── requirements-gpu.txt   # CUDA-only extras (bitsandbytes) for a GPU box
├── requirements-dev.txt   # lint + test only, used by CI
└── Makefile               # setup / smoke / baseline / data / train / eval-* / test / lint
```

---

## Roadmap

- ✅ Curate a de-leaked NL→SQL **training** set, kept strictly separate from eval.
- ✅ **Fine-tune** with LoRA, and add **execution-accuracy** scoring on a seeded SQLite DB.
- ✅ Add an **out-of-template phrasing** eval, quantifying a 100% → 75% generalisation gap.
- ✅ Add a **second schema**: the fine-tune holds 100% on a schema it never trained on.
- ✅ **Close the phrasing gap** with paraphrase augmentation and a balanced pattern mix
  (75% → 90%).
- ✅ Add **multi-table `JOIN`** evals, which exposed catastrophic forgetting (base 64% → 9%),
  then teach six join families: **100% on both** join sets, including a join key that appears
  nowhere in training.
- ✅ **Teach the magnitude words**: "total headcount" now returns `COUNT(*)`, taught by a
  contrastive `SUM` pattern rather than by memorising the wording.
- ✅ Get a genuinely **blind** estimate: a held-back set of 24 fresh intents, written after
  the model was frozen and scored once - **29% → 75%**, against 98% on the development sets.
- ⬜ Retire and replace the blind set. It has been spent, so the next unbiased measurement
  needs a fresh one, ideally written by someone other than the person curating the data.
- ⬜ Teach the constructs the blind set found missing (`IS NULL`, `SELECT DISTINCT`, date
  functions, `LIKE`) - **only** alongside a new blind set, never to chase the old one.
- ⬜ Close the last in-taxonomy failure (`GROUP BY department` losing to a lexically adjacent
  projection).
- ⬜ Optional: quantize the model and serve it behind a small API.

---

## Appendix

**Compute & environment.** Local: Apple Silicon macOS, CPU/MPS - used for scaffolding,
training and evaluation. A cloud **T4** (Colab / Kaggle free tier) is the fallback for larger
runs; `bitsandbytes` / 4-bit QLoRA is **CUDA-only**, so it lives in `requirements-gpu.txt` and
is not installed locally.

**Tech stack.** Python · PyTorch · Hugging Face `transformers`, `datasets`, `peft` (LoRA),
`accelerate` (+ `bitsandbytes` on a GPU). Experiment tracking: JSON/CSV now, Weights & Biases
optional. CI runs ruff + pytest on the deterministic core only - no torch, no model download.

**Evaluation principles.** Measure a baseline *before* training, or improvement cannot be
proven. Watch eval loss, not just train loss. Keep tokenization and prompt format identical
between training and evaluation. Never claim a result that is not on a held-out set.

**Scope (tracks).** *Track A - LoRA fine-tune a small open model (current):* post-training and
evaluation on a focused task, cheap enough for a free GPU. *Track B - pre-train a tiny GPT
from scratch (stretch, nanoGPT-style):* to exercise the pre-training loop itself on a compact
corpus.

**Why this project.** An end-to-end demonstration of small-LLM post-training and honest
evaluation - data curation, LoRA fine-tuning, memory troubleshooting, and a rigorous
before/after comparison on held-out sets - rather than calling a hosted API.
