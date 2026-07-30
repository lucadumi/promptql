# Eval set: text-to-SQL (held-out)

This is the **held-out evaluation set** used for the before/after story. It is the
same file for the baseline (Week 0) and the fine-tuned model (Week 3), so the
numbers are directly comparable.

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
  `ORDER BY`, `LIMIT`, `DISTINCT`, `GROUP BY`, and `HAVING`.
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

## Important
- This is the **eval** split only. Keep training data (Week 1) in a separate file
  (e.g. `data/train/…`) and do **not** let eval questions leak into training.
