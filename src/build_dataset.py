"""Build the NL->SQL *training* set by template-based synthetic generation.

Why generate instead of hand-write?
  Fine-tuning a model to memorise 20 answers proves nothing. We want a *training*
  set that teaches the same SQL *skills* the eval set tests (projection, COUNT/
  AVG/SUM/MAX/MIN, WHERE on numbers/strings/dates, ORDER BY, LIMIT, DISTINCT,
  GROUP BY, HAVING) but with *different questions and values*, so improvement on
  the held-out eval reflects genuine generalisation.

The three rules this script enforces (the honest-evaluation contract):
  1. SAME schema + SAME canonical SQL style as `data/eval` (see src/data_utils.py
     and src/metrics.py) so train/eval prompt+target formats are identical.
  2. NO LEAKAGE: drop any generated pair whose (normalised) question OR
     (normalised) SQL collides with an eval example. The SQL check reuses
     `src.metrics.normalize_sql` -- the *exact* function used for scoring -- so a
     training target can never equal a graded eval answer string.
  3. Reproducible: everything is seeded, so re-running yields the same split.

Output (JSONL, identical record schema to the eval file -> reuses load_jsonl):
    data/train/text2sql_train.jsonl
    data/train/text2sql_val.jsonl     # stratified 10% held out to watch eval loss

Run it (from the repo root, inside the venv):
    python -m src.build_dataset                 # write train/val + print a report
    python -m src.build_dataset --val-frac 0.15 # bigger validation split
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

# Make `src` importable whether run as `-m src.build_dataset` or as a file path.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_utils import load_jsonl  # noqa: E402
from src.metrics import normalize_sql  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_EVAL = REPO_ROOT / "data" / "eval" / "text2sql_eval.jsonl"
DEFAULT_OUTDIR = REPO_ROOT / "data" / "train"

# A generated example carries a `category` tag (the template it came from). The
# tag is used only for the stratified split + the report; it is NOT written to
# the JSONL, which keeps the on-disk schema identical to the eval file.
Example = Tuple[str, str, str]  # (category, question, sql)

# ---------------------------------------------------------------------------
# Parameter pools. We deliberately avoid the exact literals used in the eval set
# (salary 100000, budget 500000, date 2020-01-01, top-5, having-10) to spend
# fewer candidates on pairs the leakage filter would only throw away. The filter
# is still the real guarantee -- these lists are just an efficiency nicety.
# ---------------------------------------------------------------------------
DEPARTMENTS = [
    "Engineering", "Sales", "Marketing", "Finance", "Human Resources",
    "Operations", "Support", "Research", "Legal", "Product", "Design", "IT",
]
LOCATIONS = [
    "New York", "San Francisco", "London", "Berlin", "Tokyo", "Austin",
    "Seattle", "Boston", "Chicago", "Remote", "Paris", "Toronto",
]
SALARY_THRESHOLDS = [40000, 50000, 60000, 70000, 80000, 90000,
                     110000, 120000, 130000, 150000, 175000, 200000]
BUDGET_THRESHOLDS = [100000, 200000, 250000, 300000, 400000,
                     600000, 750000, 800000, 1000000, 1500000]
HIRE_DATES = ["2017-01-01", "2018-01-01", "2018-06-01", "2019-01-01",
              "2019-06-01", "2021-01-01", "2021-06-01", "2022-01-01",
              "2022-06-01", "2023-01-01"]
TOP_N = [3, 5, 10, 15]
HAVING_N = [3, 5, 15, 20, 25]

AGGS = {"AVG": "average", "MAX": "highest", "MIN": "lowest", "SUM": "total"}


def generate_candidates() -> List[Example]:
    """Expand every template over its parameter pool into (category, q, sql)."""
    out: List[Example] = []

    def add(cat: str, q: str, sql: str) -> None:
        out.append((cat, q, sql))

    # -- projection: single column from a table -----------------------------
    for phrase, col in [("salaries", "salary"),
                        ("hire dates", "hire_date")]:
        add("project_emp", f"List the {phrase} of all employees.",
            f"SELECT {col} FROM employees")
    add("project_emp", "Show the department of each employee.",
        "SELECT department FROM employees")
    add("project_emp", "What are the names of all employees?",
        "SELECT name FROM employees")
    for phrase, col in [("names", "name"), ("budgets", "budget"),
                        ("locations", "location")]:
        add("project_dept", f"List the {phrase} of all departments.",
            f"SELECT {col} FROM departments")

    # -- SELECT * filtered by department ------------------------------------
    for dept in DEPARTMENTS:
        add("select_star_dept", f"Show all employees in the {dept} department.",
            f"SELECT * FROM employees WHERE department = '{dept}'")

    # -- COUNT(*) whole table (+ paraphrase) --------------------------------
    add("count_all", "How many employees are there?",
        "SELECT COUNT(*) FROM employees")
    add("count_all", "How many departments are there?",
        "SELECT COUNT(*) FROM departments")
    add("count_all", "Count the total number of departments.",
        "SELECT COUNT(*) FROM departments")

    # -- COUNT(*) filtered by department (+ paraphrase) ---------------------
    for dept in DEPARTMENTS:
        add("count_dept", f"How many employees are in the {dept} department?",
            f"SELECT COUNT(*) FROM employees WHERE department = '{dept}'")
    for dept in DEPARTMENTS[:6]:
        add("count_dept", f"Count the employees who work in the {dept} department.",
            f"SELECT COUNT(*) FROM employees WHERE department = '{dept}'")

    # -- aggregates over salary (all employees) -----------------------------
    for agg, word in AGGS.items():
        add("agg_salary", f"What is the {word} salary of all employees?",
            f"SELECT {agg}(salary) FROM employees")
    add("agg_salary", "Find the highest salary among all employees.",
        "SELECT MAX(salary) FROM employees")
    add("agg_salary", "Find the lowest salary among all employees.",
        "SELECT MIN(salary) FROM employees")

    # -- aggregates over budget (all departments) ---------------------------
    for agg, word in AGGS.items():
        add("agg_budget", f"What is the {word} budget across all departments?",
            f"SELECT {agg}(budget) FROM departments")

    # -- WHERE on a numeric threshold (salary), both directions -------------
    for n in SALARY_THRESHOLDS:
        add("where_salary", f"List the names of employees who earn more than {n}.",
            f"SELECT name FROM employees WHERE salary > {n}")
        add("where_salary", f"List the names of employees who earn less than {n}.",
            f"SELECT name FROM employees WHERE salary < {n}")
    for n in SALARY_THRESHOLDS[:6]:
        add("where_salary", f"Show the names of employees with a salary above {n}.",
            f"SELECT name FROM employees WHERE salary > {n}")

    # -- WHERE on a date (hire_date), both directions -----------------------
    for d in HIRE_DATES:
        add("where_date", f"List the names of employees hired after {d}.",
            f"SELECT name FROM employees WHERE hire_date > '{d}'")
        add("where_date", f"List the names of employees hired before {d}.",
            f"SELECT name FROM employees WHERE hire_date < '{d}'")

    # -- ORDER BY on a column, both directions ------------------------------
    for col in ["salary", "name", "hire_date"]:
        for word, direction in [("descending", "DESC"), ("ascending", "ASC")]:
            add("order_by",
                f"Show the names of employees ordered by {col} in {word} order.",
                f"SELECT name FROM employees ORDER BY {col} {direction}")

    # -- COUNT(DISTINCT ...) ------------------------------------------------
    add("distinct", "How many distinct departments are there in the employees table?",
        "SELECT COUNT(DISTINCT department) FROM employees")
    add("distinct", "How many distinct locations are there in the departments table?",
        "SELECT COUNT(DISTINCT location) FROM departments")

    # -- departments filtered by budget threshold, both directions ----------
    for n in BUDGET_THRESHOLDS:
        add("where_budget", f"Show all departments with a budget over {n}.",
            f"SELECT * FROM departments WHERE budget > {n}")
        add("where_budget", f"Show all departments with a budget under {n}.",
            f"SELECT * FROM departments WHERE budget < {n}")

    # -- department filter + ORDER BY name ----------------------------------
    for dept in DEPARTMENTS:
        add("filter_order",
            f"List the names of employees in the {dept} department ordered by name.",
            f"SELECT name FROM employees WHERE department = '{dept}' ORDER BY name")

    # -- aggregate salary within a department -------------------------------
    for dept in DEPARTMENTS:
        add("agg_in_dept", f"What is the average salary in the {dept} department?",
            f"SELECT AVG(salary) FROM employees WHERE department = '{dept}'")
    for dept in DEPARTMENTS[:6]:
        add("agg_in_dept", f"What is the highest salary in the {dept} department?",
            f"SELECT MAX(salary) FROM employees WHERE department = '{dept}'")

    # -- departments COUNT by location --------------------------------------
    for loc in LOCATIONS:
        add("count_location", f"How many departments are located in {loc}?",
            f"SELECT COUNT(*) FROM departments WHERE location = '{loc}'")

    # -- single extreme row (ORDER BY ... LIMIT 1) --------------------------
    add("extreme_one", "Find the name of the employee with the highest salary.",
        "SELECT name FROM employees ORDER BY salary DESC LIMIT 1")
    add("extreme_one", "Find the name of the employee with the lowest salary.",
        "SELECT name FROM employees ORDER BY salary ASC LIMIT 1")

    # -- top-N (ORDER BY ... LIMIT n), both directions ----------------------
    for n in TOP_N:
        add("top_n", f"List the names of the top {n} highest paid employees.",
            f"SELECT name FROM employees ORDER BY salary DESC LIMIT {n}")
        add("top_n", f"List the names of the {n} lowest paid employees.",
            f"SELECT name FROM employees ORDER BY salary ASC LIMIT {n}")

    # -- GROUP BY with an aggregate (per department) ------------------------
    add("group_by", "Show each department and the number of employees in it.",
        "SELECT department, COUNT(*) FROM employees GROUP BY department")
    for agg, phrase in [("SUM", "total salary paid"),
                        ("AVG", "average salary"),
                        ("MAX", "highest salary")]:
        add("group_by", f"For each department, show the {phrase}.",
            f"SELECT department, {agg}(salary) FROM employees GROUP BY department")

    # -- GROUP BY ... HAVING, both directions -------------------------------
    for n in HAVING_N:
        add("having", f"List departments that have more than {n} employees.",
            f"SELECT department FROM employees GROUP BY department "
            f"HAVING COUNT(*) > {n}")
        add("having", f"List departments that have fewer than {n} employees.",
            f"SELECT department FROM employees GROUP BY department "
            f"HAVING COUNT(*) < {n}")

    return out


def normalize_question(q: str) -> str:
    """Lightweight question canonicaliser for dedup + leakage checks."""
    return " ".join(q.lower().split()).rstrip("?.").strip()


def build(eval_file: Path, val_frac: float, seed: int) -> Dict[str, object]:
    """Generate, dedup, de-leak, and stratified-split. Returns a report dict."""
    rng = random.Random(seed)

    # Leakage blocklists from the held-out eval set.
    eval_rows = load_jsonl(eval_file)
    eval_questions = {normalize_question(r["question"]) for r in eval_rows}
    eval_sqls = {normalize_sql(r["sql"]) for r in eval_rows}

    candidates = generate_candidates()

    kept: List[Example] = []
    seen_questions: set[str] = set()
    dropped_leak = 0
    dropped_dup = 0
    for cat, q, sql in candidates:
        nq, nsql = normalize_question(q), normalize_sql(sql)
        if nq in eval_questions or nsql in eval_sqls:   # rule 2: no leakage
            dropped_leak += 1
            continue
        if nq in seen_questions:                        # de-duplicate questions
            dropped_dup += 1
            continue
        seen_questions.add(nq)
        kept.append((cat, q, sql))

    # Stratified split: hold out ~val_frac of EACH category so validation
    # mirrors every SQL pattern (otherwise a rare pattern might be train-only).
    by_cat: Dict[str, List[Example]] = defaultdict(list)
    for ex in kept:
        by_cat[ex[0]].append(ex)

    train: List[Example] = []
    val: List[Example] = []
    for cat in sorted(by_cat):
        items = by_cat[cat][:]
        rng.shuffle(items)
        n_val = min(len(items) - 1, math.ceil(len(items) * val_frac)) if len(items) > 1 else 0
        val.extend(items[:n_val])
        train.extend(items[n_val:])

    rng.shuffle(train)
    rng.shuffle(val)

    return {
        "train": train,
        "val": val,
        "n_candidates": len(candidates),
        "n_kept": len(kept),
        "dropped_leak": dropped_leak,
        "dropped_dup": dropped_dup,
        "by_cat": {c: len(v) for c, v in sorted(by_cat.items())},
        "eval_questions": eval_questions,
        "eval_sqls": eval_sqls,
    }


def write_jsonl(rows: List[Example], path: Path) -> None:
    """Write clean {id, question, sql} records (schema identical to eval)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for i, (_cat, q, sql) in enumerate(rows, start=1):
            fh.write(json.dumps({"id": i, "question": q, "sql": sql}) + "\n")


def verify_no_leakage(rows: List[Example], eval_questions: set, eval_sqls: set) -> None:
    """Assert (belt and braces) that nothing written collides with eval."""
    for _cat, q, sql in rows:
        assert normalize_question(q) not in eval_questions, f"LEAK question: {q}"
        assert normalize_sql(sql) not in eval_sqls, f"LEAK sql: {sql}"


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the NL->SQL training set.")
    parser.add_argument("--eval-file", default=str(DEFAULT_EVAL))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUTDIR))
    parser.add_argument("--val-frac", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=13)
    args = parser.parse_args()

    rep = build(Path(args.eval_file), args.val_frac, args.seed)
    train, val = rep["train"], rep["val"]

    verify_no_leakage(train + val, rep["eval_questions"], rep["eval_sqls"])

    out_dir = Path(args.out_dir)
    train_path = out_dir / "text2sql_train.jsonl"
    val_path = out_dir / "text2sql_val.jsonl"
    write_jsonl(train, train_path)
    write_jsonl(val, val_path)

    print("=" * 64)
    print("PromptQL training-set build")
    print("=" * 64)
    print(f"candidates generated : {rep['n_candidates']}")
    print(f"dropped (eval leak)  : {rep['dropped_leak']}")
    print(f"dropped (duplicate)  : {rep['dropped_dup']}")
    print(f"kept (unique, clean) : {rep['n_kept']}")
    print(f"  -> train           : {len(train)}")
    print(f"  -> val             : {len(val)}")
    print("leakage vs eval      : 0 (verified)")
    print("-" * 64)
    print("per-pattern (category) counts among kept examples:")
    train_cats = Counter(c for c, _, _ in train)
    val_cats = Counter(c for c, _, _ in val)
    for cat in sorted(rep["by_cat"]):
        print(f"  {cat:16s} total={rep['by_cat'][cat]:3d}  "
              f"train={train_cats.get(cat,0):3d}  val={val_cats.get(cat,0):2d}")
    print("-" * 64)
    print(f"wrote {train_path.relative_to(REPO_ROOT)}")
    print(f"wrote {val_path.relative_to(REPO_ROOT)}")
    print("=" * 64)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
