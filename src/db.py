"""Seed SQLite database + execution-accuracy comparison for the text-to-SQL eval.

Normalised exact-match (``src/metrics.py``) is a strict *lower bound*: a
semantically-correct query written differently -- ``COUNT(*)`` vs ``COUNT(id)``,
``salary > 100000`` vs ``salary >= 100001``, an extra ``ORDER BY ... ASC`` -- is
graded wrong. The honest upgrade the README promised is *execution accuracy*: run
the predicted and gold SQL against a real SQLite database seeded with
representative rows, then compare the returned result sets. A query that returns
the right rows is credited regardless of how it was written.

This module owns two things:

  * a small, deterministic, committed seed dataset for the fixed schema, chosen so
    every held-out eval query returns a *discriminating* result (a wrong query
    returns different rows -- see the notes on the seed constants below), and
  * ``execution_match()``, which runs prediction vs gold and compares rows. Row
    order only matters when the gold query uses ``ORDER BY`` (matching the Spider
    test-suite convention); otherwise results are compared as multisets.

The schema itself is imported from ``src/data_utils.py`` so training, evaluation,
and this database can never drift apart.
"""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

from src.data_utils import SCHEMA_SQL
from src.metrics import extract_sql

Row = Tuple[object, ...]

# ---------------------------------------------------------------------------
# Seed data. Kept explicit (not RNG-generated) so the exact rows every eval
# number depends on are visible and reviewable in source control.
#
# departments: (id, name, budget, location)
#   - two departments are in 'New York'      -> COUNT(location='New York') = 2
#   - two departments have budget > 500000   -> budget filter returns 2 rows
# ---------------------------------------------------------------------------
DEPARTMENTS_SEED: List[Tuple[int, str, int, str]] = [
    (1, "Engineering", 900000, "New York"),
    (2, "Sales",       600000, "New York"),
    (3, "Marketing",   400000, "London"),
    (4, "Finance",     300000, "Berlin"),
]

# employees: (id, name, department, salary, hire_date, manager_id)
#   - salaries are ALL DISTINCT, so ORDER BY salary and every LIMIT / "top N" /
#     MIN / MAX query is deterministic (no ties -> no ambiguous result set).
#   - Engineering has 11 employees (> 10) so the HAVING COUNT(*) > 10 eval query
#     returns exactly one department; every other department has <= 4.
#   - the 11 Engineering salaries are all > 100000 and the other 9 are <= 100000,
#     so the "salary > 100000" filter is non-trivial (11 of 20 rows).
#   - hire_date straddles 2020-01-01 in both directions so the date filter is
#     discriminating rather than all-or-nothing.
EMPLOYEES_SEED: List[Tuple[int, str, str, int, str, Optional[int]]] = [
    # Engineering (11) -- all earn > 100000; Alice is the lead (no manager).
    (1,  "Alice",   "Engineering", 150000, "2021-03-01", None),
    (2,  "Bob",     "Engineering", 145000, "2019-07-15", 1),
    (3,  "Carol",   "Engineering", 140000, "2022-01-10", 1),
    (4,  "Dave",    "Engineering", 135000, "2018-05-20", 1),
    (5,  "Eve",     "Engineering", 130000, "2021-11-01", 1),
    (6,  "Frank",   "Engineering", 125000, "2020-06-30", 1),
    (7,  "Grace",   "Engineering", 120000, "2019-02-28", 1),
    (8,  "Heidi",   "Engineering", 115000, "2022-08-14", 1),
    (9,  "Ivan",    "Engineering", 110000, "2023-01-05", 1),
    (10, "Judy",    "Engineering", 105000, "2020-03-15", 1),
    (11, "Mallory", "Engineering", 101000, "2018-09-01", 1),
    # Sales (4) -- Niaj is the lead.
    (12, "Niaj",    "Sales",        99000, "2021-04-01", None),
    (13, "Olivia",  "Sales",        95000, "2019-12-01", 12),
    (14, "Peggy",   "Sales",        90000, "2022-05-05", 12),
    (15, "Rupert",  "Sales",        85000, "2020-10-20", 12),
    # Marketing (3) -- Sybil is the lead.
    (16, "Sybil",   "Marketing",    80000, "2021-01-15", None),
    (17, "Trent",   "Marketing",    75000, "2018-03-03", 16),
    (18, "Victor",  "Marketing",    70000, "2023-02-02", 16),
    # Finance (2) -- Walter is the lead; Xavier has the unique lowest salary.
    (19, "Walter",  "Finance",      65000, "2019-08-08", None),
    (20, "Xavier",  "Finance",      60000, "2020-12-25", 19),
]


def build_db(conn: Optional[sqlite3.Connection] = None) -> sqlite3.Connection:
    """Create the schema and insert the seed rows.

    Defaults to a fresh in-memory database; pass a connection to seed an existing
    one (e.g. a file-backed DB). Returns the connection for chaining.
    """
    if conn is None:
        conn = sqlite3.connect(":memory:")
    conn.executescript(SCHEMA_SQL)
    conn.executemany(
        "INSERT INTO departments (id, name, budget, location) VALUES (?, ?, ?, ?)",
        DEPARTMENTS_SEED,
    )
    conn.executemany(
        "INSERT INTO employees (id, name, department, salary, hire_date, manager_id) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        EMPLOYEES_SEED,
    )
    conn.commit()
    return conn


def run_sql(conn: sqlite3.Connection, sql: str) -> List[Row]:
    """Execute one SQL statement and return the fetched rows as a list of tuples."""
    return conn.execute(sql).fetchall()


@dataclass
class ExecResult:
    """Outcome of comparing a predicted query against the gold query by execution.

    ``match`` is the headline signal. ``pred_error`` / ``gold_error`` hold the
    SQLite error message when a query failed to run (useful for debugging model
    output), and are ``None`` on success.
    """

    match: bool
    pred_error: Optional[str] = None
    gold_error: Optional[str] = None


def _is_ordered(sql: str) -> bool:
    """True if the (gold) query pins a row order, i.e. contains ORDER BY."""
    return "order by" in sql.lower()


def _norm_value(value: object) -> object:
    """Round floats so equivalent aggregates don't differ by float noise."""
    if isinstance(value, float):
        return round(value, 6)
    return value


def _canonical(rows: Sequence[Row], ordered: bool) -> List[str]:
    """Represent a result set for comparison.

    Rows are rendered to ``repr`` strings (after light float rounding) so mixed
    column types never raise while sorting. When the gold query is unordered we
    sort -- comparing as a multiset; when it is ordered we keep the row order.
    """
    reprs = [repr(tuple(_norm_value(v) for v in row)) for row in rows]
    return reprs if ordered else sorted(reprs)


def execution_match(pred_raw: str, gold_sql: str) -> ExecResult:
    """Run prediction vs gold on a fresh seeded DB and compare their result sets.

    ``pred_raw`` is the model's raw completion; the runnable SQL is extracted from
    it first (stripping markdown fences / leading prose, as in exact-match). A
    prediction that yields no SQL, or that raises a SQLite error, is a non-match
    with the reason recorded in the returned :class:`ExecResult`.
    """
    conn = build_db()
    try:
        try:
            gold_rows = run_sql(conn, gold_sql)
        except sqlite3.Error as exc:  # a broken gold is a data bug, not a pass
            return ExecResult(False, gold_error=str(exc))

        pred_sql = extract_sql(pred_raw)
        if not pred_sql:
            return ExecResult(False, pred_error="no SQL found in prediction")
        try:
            pred_rows = run_sql(conn, pred_sql)
        except sqlite3.Error as exc:
            return ExecResult(False, pred_error=str(exc))

        ordered = _is_ordered(gold_sql)
        match = _canonical(pred_rows, ordered) == _canonical(gold_rows, ordered)
        return ExecResult(match)
    finally:
        conn.close()
