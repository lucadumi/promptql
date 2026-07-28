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

## Fixed schema
Every question is written against this SQLite schema (also shown to the model in
every prompt, see `src/data_utils.py::SCHEMA_SQL`):

```sql
CREATE TABLE departments (id INTEGER PRIMARY KEY, name TEXT, budget INTEGER, location TEXT);
CREATE TABLE employees  (id INTEGER PRIMARY KEY, name TEXT, department TEXT,
                         salary INTEGER, hire_date TEXT, manager_id INTEGER);
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

## Important
- This is the **eval** split only. Keep training data (Week 1) in a separate file
  (e.g. `data/train/…`) and do **not** let eval questions leak into training.
