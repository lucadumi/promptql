# Eval sets: text-to-SQL (held-out)

The **held-out evaluation data** behind the before/after story. The same files are used
for the baseline and for every fine-tuned model, so the numbers are directly comparable.

They fall into two groups, and the distinction matters more than any individual file:

- **Six development sets.** Each isolates one variable, and each has fed at least one
  data-curation decision. They are honest measurements of what they measure, but they are
  no longer *unbiased* - failures on them were read and acted upon.
- **One held-back (blind) set.** Written by an independent author who never saw the
  training generator, the other eval sets or a single model prediction; scored once; never
  used to steer the training data. It is the project's only unbiased estimate, and it stays
  that way only while it is left alone.

A blind set is spendable, and this project has spent one. `text2sql_eval_blind_v1_retired.jsonl`
*was* the blind set. Its failures were read, the construct families they exposed were taught,
and that is precisely what converts a blind set into development signal. So it was retired
into the development group - renamed, added to `make eval-all`, and no longer quoted as
unbiased - and replaced by `text2sql_eval_blind_v2.jsonl`. Retiring it is the honest
accounting; deleting it would lose a useful regression set, and re-scoring it as though it
were still blind would be a lie.

## Files
- `text2sql_eval.jsonl` - 20 examples, one JSON object per line:
  ```json
  {"id": 1, "question": "List the names of all employees.", "sql": "SELECT name FROM employees"}
  ```
- `text2sql_eval_paraphrase.jsonl` - the **out-of-template** set: the *same 20 gold
  queries* as above, but every question is reworded in unfamiliar, indirect language.
  Same schema, so it reuses the same prompt and seeded DB. Used to measure robustness to
  unseen phrasings (see `tests/test_eval_sets.py`, which checks the golds match and that
  no question overlaps a training example).
- `text2sql_eval_bookstore.jsonl` - the **cross-schema** set: the same 20 intents and
  construct coverage, but re-targeted at a completely different **bookstore** schema
  (`publishers`, `books`) with all-new table and column names. Different schema, so it uses
  the bookstore prompt and its own seeded DB (`src/db.py::BOOKSTORE_SCHEMA`). Used to measure
  whether the fine-tune transfers to a schema it never trained on (see
  `tests/test_schema_bookstore.py`).
- `text2sql_eval_join.jsonl` - the **multi-table JOIN** set: 12 questions that cannot be
  answered from one table. Eleven join `employees` to `departments` on the schema's TEXT key
  (`employees.department = departments.name`) and one is a **self-join** through
  `manager_id`. Employees schema, same seeded DB. Used to measure a construct that no other
  eval set contains at all (see `tests/test_eval_join.py`).
- `text2sql_eval_join_bookstore.jsonl` - the **cross-schema JOIN** set: the same 11
  join intents (the self-join has no counterpart, see below) re-asked on the bookstore
  schema, where the two tables are related by an **integer foreign key**
  (`books.publisher_id = publishers.id`) instead of a text name. Bookstore prompt and DB.
  Training only ever shows the text-key join, so this set measures whether the model
  transfers *the idea of a join* or memorised one join condition.
- `text2sql_eval_blind_v1_retired.jsonl` - the **retired first blind set**: 24 fresh intents
  on the employees schema, written after the first shipped adapter was frozen and scored
  once. Its six failures showed that four of them needed a construct present in **no**
  training template (`strftime`, a bare `SELECT DISTINCT`, `IS NULL`, de-duplicating a join
  result). Teaching those families spent this set, so it now lives in the development group
  and runs as part of `make eval-all`, where it guards against regressing them.
- `text2sql_eval_blind_v2.jsonl` - the **current held-back (blind)** set: 30 fresh intents
  on the employees schema, tagged `easy` / `medium` / `hard` by their author. Written by an
  independent party under enforced isolation (see below). Scored **once** per model; see
  `tests/test_eval_blind.py`.


## Fixed schema
Every question is written against this SQLite schema (also shown to the model in
every prompt, see `src/data_utils.py::SCHEMA_SQL`):

```sql
CREATE TABLE departments (id INTEGER PRIMARY KEY, name TEXT, budget INTEGER, location TEXT);
CREATE TABLE employees  (id INTEGER PRIMARY KEY, name TEXT, department TEXT,
                         salary INTEGER, hire_date TEXT, manager_id INTEGER);
```

The **cross-schema** set (`text2sql_eval_bookstore.jsonl`) targets a second schema that
mirrors the one above construct-for-construct with all-new names (also shown to the model in
every prompt, see `src/data_utils.py::BOOKSTORE_SCHEMA_SQL`):

```sql
CREATE TABLE publishers (id INTEGER PRIMARY KEY, name TEXT, revenue INTEGER, city TEXT);
CREATE TABLE books      (id INTEGER PRIMARY KEY, title TEXT, genre TEXT,
                         price INTEGER, published_date TEXT, publisher_id INTEGER);
```

## How it was built (data curation notes)
- Hand-written by me to cover a spread of SQL constructs:
  projection, `COUNT`/`AVG`/`SUM`/`MAX`, `WHERE` (numeric, string, date),
  `ORDER BY`, `LIMIT`, `DISTINCT`, `GROUP BY`, and `HAVING`. The one exception is
  `text2sql_eval_blind_v2.jsonl`, which was written by an independent author precisely so
  that the project's headline number does not depend on the data curator also setting the
  exam - see "The blind set specifically" below.
- Gold SQL is canonical and minimal (no trailing semicolon, single quotes for
  string literals) so it matches the normalisation in `src/metrics.py`.
- Scored two ways: **normalised exact-match** (strict string equality - a
  conservative lower bound) and **execution accuracy** (run predicted vs. gold SQL
  against the seeded SQLite DB in `src/db.py` and compare the returned rows), which
  credits equivalent-but-differently-written queries.

### The JOIN sets specifically
- **Every gold really needs the join.** Filters and groupings use a column that only
  exists in the *other* table (`departments.location`, `publishers.city`), so a
  single-table query returns different rows and cannot score by accident. The seed
  data keeps them discriminating: the 4 departments sit in 3 locations, the highest
  salary in London (80000) is not the global highest (150000), and 4 of the 20
  employees have no manager, so the inner self-join returns 16 rows rather than 20.
  `tests/test_eval_join.py::TestJoinIsNecessary` asserts each of these.
- **Two different join keys, on purpose.** The employees schema is denormalised, so
  its join is on a TEXT name; the bookstore schema uses a conventional integer FK.
  The training set only ever teaches the first, which makes the bookstore set a
  transfer measurement (and it is: the base model is far *better* at the FK join
  than at the text-key one).
- **The self-join has no cross-schema mirror** - `books` has no self-referencing
  column - so the bookstore set re-asks 11 of the 12 intents, keeping the same ids.
- Column names in join golds are fully qualified (`employees.name`,
  `departments.budget`); aliases are used only where the query needs them (the
  self-join). Execution accuracy is what makes alias style irrelevant to scoring.

### The blind set specifically

- **Written by someone else, under enforced isolation.** The weakest part of v1 was that
  the person who curated the training data also wrote the "unbiased" test. v2 fixes that: it
  was authored by an independent agent given read access to exactly two files -
  `src/data_utils.py` (the schema) and `src/db.py` (the seed rows) - and explicitly denied
  the training generator, every existing eval set, the results directory, the READMEs and
  any model output. It was told to write the questions a data analyst would actually ask,
  with a spread of difficulty, and was *not* told which SQL constructs the project teaches.
- **The author owns the questions; the curator only fixed well-posedness.** Two rounds of
  review changed golds, never intents: cosmetic `ORDER BY` clauses were removed (see below),
  and questions that did not say which columns to return were made explicit. Four questions
  were then copy-edited for English fluency, preserving the author's semantics exactly. All
  of this happened *before* any model was run.
- **Ordering has to be earned.** `execution_match` compares result sets as a multiset unless
  the gold contains `ORDER BY`, in which case row order must match exactly. A gold that
  orders for readability therefore grades a correct answer wrong. Every `ORDER BY` in this
  set is justified by the question ("highest first", "cheapest first") or is load-bearing for
  a `LIMIT`; `tests/test_eval_blind.py::TestOrderingIsEarned` enforces it.
- **No question is reachable by the generator; four golds are.** No blind *question* can be
  produced by `src/build_dataset.py` under any phrasing or parameter - so none is a training
  template in disguise. Four *golds* are reachable, because questions like "what's the
  average salary in Sales?" have exactly one natural SQL answer. Rather than distort the eval
  set to dodge that, the leakage filter deletes those targets from training entirely (13
  candidate rows), which is the same treatment every other eval gold gets: the model has to
  reach them by generalising. This is the one property where v2 is weaker than v1, and it is
  a deliberate trade - v1's stricter version only mattered because it was scored against an
  already-frozen adapter without retraining.
- **It is graded per difficulty tier.** The author tagged each question `easy`, `medium` or
  `hard`, and the results are reported per tier. An aggregate over a set that ranges from
  `SELECT salary FROM employees WHERE name = 'Peggy'` to correlated `NOT EXISTS` subqueries
  hides which half of the distribution the model actually fails on.
- **Selectivity is enforced.** No single-table gold may return all 20 employees, so a
  degenerate `SELECT ... FROM employees` cannot score by accident.
- **The protocol is the point.** `make eval-blind` is deliberately excluded from
  `make eval-all`, and `tests/test_eval_blind.py` fails if anyone adds it. Score a model
  once, record it, and do **not** curate against the failures. The moment a failure here is
  "fixed", this becomes just another dev set - which is exactly what happened to v1, and why
  v1 is now named `_retired` and sits in `eval-all`.

## Important
- This is the **eval** split only. Keep training data in a separate file
  (e.g. `data/train/…`) and do **not** let eval questions leak into training.
- The blind set carries one extra rule on top of that: do not read its failures as a to-do
  list. Fixing them is exactly what would destroy its value. If its findings are worth
  acting on, retire it first - rename it, move it into `eval-all`, and commission a
  replacement - so that the cost of acting is paid openly instead of hidden.
