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
and this database can never drift apart. A second, unrelated ``bookstore`` schema
lives here too (same two-table shape, all-new names) purely for the cross-schema
generalisation eval; :class:`Schema` bundles a schema's DDL with its seed rows so
the right database is built for whichever eval set is being scored.
"""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence, Tuple

from src.data_utils import BOOKSTORE_SCHEMA_SQL, SCHEMA_SQL
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
#   - JOIN invariants (data/eval/text2sql_eval_join.jsonl): every employee's
#     department matches a departments.name, so the inner join keeps all 20 rows;
#     the 4 departments sit in only 3 locations (2x New York), so grouping by
#     location is not grouping by department; New York holds 15 employees (> 10)
#     while no other location does; the highest London salary (80000) and the
#     highest Berlin salary (65000) both differ from the global 150000; and the 4
#     department leads have manager_id NULL, so the manager self-join returns 16
#     of 20 rows (a LEFT JOIN would return 20).
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

# ---------------------------------------------------------------------------
# Bookstore seed data for the cross-schema eval. Deliberately mirrors the
# employees invariants one-for-one so the same eval constructs stay
# discriminating on a schema the model never trained on:
#   - publishers: two are in 'New York' and two have revenue > 500000.
#   - books: all 20 prices are DISTINCT (deterministic ORDER BY / LIMIT / MIN /
#     MAX); the 'Fiction' genre has 11 books (> 10) for the HAVING query while
#     every other genre has <= 4; those same 11 Fiction books are the only ones
#     priced > 30; and published_date straddles 2015-01-01 in both directions.
#   - JOIN invariants (data/eval/text2sql_eval_join_bookstore.jsonl): publisher_id
#     is a real integer FK (never NULL), so the inner join keeps all 20 books; the
#     4 publishers sit in 3 cities, with New York holding 12 books (> 10) and no
#     other city close; and the top London price (48) and top Berlin price (39)
#     both differ from the global 60.
# ---------------------------------------------------------------------------
PUBLISHERS_SEED: List[Tuple[int, str, int, str]] = [
    (1, "Penguin",       900000, "New York"),
    (2, "HarperCollins", 600000, "New York"),
    (3, "Oxford Press",  400000, "London"),
    (4, "Cedar Books",   300000, "Berlin"),
]

# books: (id, title, genre, price, published_date, publisher_id)
BOOKS_SEED: List[Tuple[int, str, str, int, str, int]] = [
    # Fiction (11) -- the only books priced > 30; prices all distinct.
    (1,  "The Silent River",  "Fiction", 60, "2016-04-01", 1),
    (2,  "Northern Lights",   "Fiction", 57, "2014-07-15", 1),
    (3,  "A Distant Shore",   "Fiction", 54, "2018-01-10", 2),
    (4,  "The Last Summer",   "Fiction", 51, "2012-05-20", 1),
    (5,  "Winter's Tale",     "Fiction", 48, "2019-11-01", 3),
    (6,  "Echoes of Time",    "Fiction", 45, "2016-06-30", 2),
    (7,  "The Glass House",   "Fiction", 42, "2013-02-28", 1),
    (8,  "Paper Moons",       "Fiction", 39, "2020-08-14", 4),
    (9,  "Shadow and Light",  "Fiction", 36, "2017-01-05", 2),
    (10, "The Long Road",     "Fiction", 34, "2011-03-15", 1),
    (11, "Quiet Places",      "Fiction", 31, "2018-09-01", 3),
    # Science (4).
    (12, "Cosmos Explained",  "Science", 30, "2015-04-01", 2),
    (13, "The Quantum World", "Science", 28, "2014-12-01", 3),
    (14, "Genes and Us",      "Science", 25, "2021-05-05", 2),
    (15, "Climate Futures",   "Science", 22, "2019-10-20", 4),
    # History (3).
    (16, "Empires of Old",    "History", 20, "2013-01-15", 3),
    (17, "The War Years",     "History", 18, "2016-03-03", 1),
    (18, "Ancient Roads",     "History", 15, "2020-02-02", 3),
    # Poetry (2) -- Morning Songs has the unique lowest price.
    (19, "Collected Verse",   "Poetry",  12, "2012-08-08", 4),
    (20, "Morning Songs",     "Poetry",  10, "2018-12-25", 2),
]


def _seed_employees(conn: sqlite3.Connection) -> None:
    conn.executemany(
        "INSERT INTO departments (id, name, budget, location) VALUES (?, ?, ?, ?)",
        DEPARTMENTS_SEED,
    )
    conn.executemany(
        "INSERT INTO employees (id, name, department, salary, hire_date, manager_id) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        EMPLOYEES_SEED,
    )


def _seed_bookstore(conn: sqlite3.Connection) -> None:
    conn.executemany(
        "INSERT INTO publishers (id, name, revenue, city) VALUES (?, ?, ?, ?)",
        PUBLISHERS_SEED,
    )
    conn.executemany(
        "INSERT INTO books (id, title, genre, price, published_date, publisher_id) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        BOOKS_SEED,
    )


@dataclass(frozen=True)
class Schema:
    """A named database schema: the DDL shown in the prompt plus its seed rows.

    ``ddl`` is the exact ``CREATE TABLE`` text the model sees (from
    ``src/data_utils.py``); ``seed`` inserts the committed rows into a fresh
    connection. Keeping both together guarantees the prompt schema and the
    execution database can never disagree.
    """

    name: str
    ddl: str
    seed: Callable[[sqlite3.Connection], None]


EMPLOYEES_SCHEMA = Schema("employees", SCHEMA_SQL, _seed_employees)
BOOKSTORE_SCHEMA = Schema("bookstore", BOOKSTORE_SCHEMA_SQL, _seed_bookstore)

# Registry keyed by name, used by the eval CLI's --schema flag.
SCHEMAS = {s.name: s for s in (EMPLOYEES_SCHEMA, BOOKSTORE_SCHEMA)}


def build_db(
    schema: Schema = EMPLOYEES_SCHEMA, conn: Optional[sqlite3.Connection] = None
) -> sqlite3.Connection:
    """Create ``schema``'s tables and insert its seed rows.

    Defaults to the employees schema in a fresh in-memory database; pass a
    connection to seed an existing one (e.g. a file-backed DB). Returns the
    connection for chaining.
    """
    if conn is None:
        conn = sqlite3.connect(":memory:")
    conn.executescript(schema.ddl)
    schema.seed(conn)
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


def execution_match(
    pred_raw: str, gold_sql: str, schema: Schema = EMPLOYEES_SCHEMA
) -> ExecResult:
    """Run prediction vs gold on a fresh seeded DB and compare their result sets.

    ``pred_raw`` is the model's raw completion; the runnable SQL is extracted from
    it first (stripping markdown fences / leading prose, as in exact-match). A
    prediction that yields no SQL, or that raises a SQLite error, is a non-match
    with the reason recorded in the returned :class:`ExecResult`. ``schema``
    selects which database to build (defaults to the employees schema).
    """
    conn = build_db(schema)
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
