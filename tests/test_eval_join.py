"""Unit tests for the multi-table JOIN eval sets.

Until now no eval set contained a JOIN, so nothing measured whether the fine-tune
could relate two tables. `data/eval/text2sql_eval_join.jsonl` adds that on the
employees schema (joined on a TEXT key, plus one self-join through manager_id),
and `data/eval/text2sql_eval_join_bookstore.jsonl` re-asks the same intents on the
bookstore schema, where the join is an INTEGER foreign key the training data never
shows. These tests lock in the properties that make those two numbers meaningful:

  * every gold really needs a join (a single-table query returns different rows),
  * every gold runs and returns rows on its own seeded DB, and
  * no join question or gold SQL leaks into the generated training set.

Pure data + SQLite, so the suite stays fast (no torch / model download).
"""
from pathlib import Path

import pytest

from src.build_dataset import normalize_question
from src.data_utils import load_jsonl
from src.db import BOOKSTORE_SCHEMA, EMPLOYEES_SCHEMA, build_db, run_sql
from src.metrics import normalize_sql

REPO_ROOT = Path(__file__).resolve().parents[1]
JOIN = REPO_ROOT / "data" / "eval" / "text2sql_eval_join.jsonl"
JOIN_BOOKSTORE = REPO_ROOT / "data" / "eval" / "text2sql_eval_join_bookstore.jsonl"
TRAIN = REPO_ROOT / "data" / "train" / "text2sql_train.jsonl"
VAL = REPO_ROOT / "data" / "train" / "text2sql_val.jsonl"

# The self-join (employees -> employees via manager_id) has no bookstore
# counterpart: books has no self-referencing column. It is therefore the one
# employees intent the cross-schema mirror cannot re-ask.
SELF_JOIN_ID = 12


@pytest.fixture(scope="module")
def join_rows():
    return load_jsonl(JOIN)


@pytest.fixture(scope="module")
def bookstore_rows():
    return load_jsonl(JOIN_BOOKSTORE)


class TestShape:
    def test_rows_are_well_formed(self, join_rows, bookstore_rows):
        assert len(join_rows) == 12
        assert len(bookstore_rows) == 11
        for r in join_rows + bookstore_rows:
            assert set(r) >= {"id", "question", "sql"}
            assert r["question"].strip()
            assert r["sql"].strip()

    def test_ids_are_unique(self, join_rows, bookstore_rows):
        for rows in (join_rows, bookstore_rows):
            ids = [r["id"] for r in rows]
            assert len(set(ids)) == len(ids)

    def test_every_gold_is_a_join(self, join_rows, bookstore_rows):
        for r in join_rows + bookstore_rows:
            assert " join " in normalize_sql(r["sql"]), f"id {r['id']} is not a join"


class TestExecutableOnSeedDb:
    def test_employees_golds_run_and_return_rows(self, join_rows):
        conn = build_db(EMPLOYEES_SCHEMA)
        try:
            for r in join_rows:
                assert run_sql(conn, r["sql"]), f"empty/failed gold {r['id']}: {r['sql']}"
        finally:
            conn.close()

    def test_bookstore_golds_run_and_return_rows(self, bookstore_rows):
        conn = build_db(BOOKSTORE_SCHEMA)
        try:
            for r in bookstore_rows:
                assert run_sql(conn, r["sql"]), f"empty/failed gold {r['id']}: {r['sql']}"
        finally:
            conn.close()


class TestConstructCoverage:
    def test_join_set_exercises_the_full_construct_range(self, join_rows):
        golds = " ".join(r["sql"].upper() for r in join_rows)
        for kw in ("JOIN", "WHERE", "GROUP BY", "HAVING", "ORDER BY", "LIMIT",
                   "COUNT(", "AVG(", "MAX("):
            assert kw in golds, f"join eval is missing construct: {kw}"

    def test_bookstore_mirror_covers_the_same_constructs(self, bookstore_rows):
        golds = " ".join(r["sql"].upper() for r in bookstore_rows)
        for kw in ("JOIN", "WHERE", "GROUP BY", "HAVING", "ORDER BY", "LIMIT",
                   "COUNT(", "AVG(", "MAX("):
            assert kw in golds, f"bookstore join eval is missing construct: {kw}"

    def test_employees_set_includes_a_self_join(self, join_rows):
        row = next(r for r in join_rows if r["id"] == SELF_JOIN_ID)
        sql = normalize_sql(row["sql"])
        assert "employees e join employees m" in sql
        assert "manager_id" in sql


class TestJoinIsNecessary:
    """A gold that a single-table query could answer would not test anything."""

    @pytest.fixture(scope="class")
    def conn(self):
        conn = build_db(EMPLOYEES_SCHEMA)
        yield conn
        conn.close()

    def _gold(self, join_rows, gold_id):
        return next(r["sql"] for r in join_rows if r["id"] == gold_id)

    def test_grouping_by_location_is_not_grouping_by_department(self, conn, join_rows):
        # 4 departments live in 3 locations, so only the join collapses them.
        joined = run_sql(conn, self._gold(join_rows, 7))
        single = run_sql(conn, "SELECT department, COUNT(*) FROM employees GROUP BY department")
        assert len(joined) == 3 and len(single) == 4

    def test_filtering_on_location_selects_a_strict_subset(self, conn, join_rows):
        joined = run_sql(conn, self._gold(join_rows, 3))
        everyone = run_sql(conn, "SELECT name FROM employees")
        assert 0 < len(joined) < len(everyone)

    def test_max_within_a_location_differs_from_the_global_max(self, conn, join_rows):
        joined = run_sql(conn, self._gold(join_rows, 10))
        assert joined != run_sql(conn, "SELECT MAX(salary) FROM employees")

    def test_top_earner_in_berlin_is_not_the_global_top_earner(self, conn, join_rows):
        joined = run_sql(conn, self._gold(join_rows, 11))
        single = run_sql(conn, "SELECT name FROM employees ORDER BY salary DESC LIMIT 1")
        assert joined != single

    def test_self_join_drops_the_employees_without_a_manager(self, conn, join_rows):
        # 4 department leads have manager_id NULL, so an inner self-join is 16/20.
        assert len(run_sql(conn, self._gold(join_rows, SELF_JOIN_ID))) == 16


class TestCrossSchemaMirror:
    def test_bookstore_mirror_reuses_the_employees_intent_ids(self, join_rows, bookstore_rows):
        mirrored = {r["id"] for r in join_rows} - {SELF_JOIN_ID}
        assert {r["id"] for r in bookstore_rows} == mirrored

    def test_bookstore_golds_only_touch_bookstore_tables(self, bookstore_rows):
        for r in bookstore_rows:
            sql = normalize_sql(r["sql"])
            assert "books" in sql and "publishers" in sql
            for leaked in ("employees", "departments", "salary", "hire_date"):
                assert leaked not in sql, f"employees-schema leak in id {r['id']}: {r['sql']}"

    def test_bookstore_joins_on_the_integer_foreign_key(self, bookstore_rows):
        # The employees schema joins on a TEXT name; the bookstore joins on an
        # integer FK. Training only ever shows the former, which is exactly what
        # makes this set a transfer test rather than a memorisation one.
        for r in bookstore_rows:
            assert "books.publisher_id = publishers.id" in normalize_sql(r["sql"])


class TestNoLeakageIntoTraining:
    @pytest.fixture(scope="class")
    def training(self):
        return load_jsonl(TRAIN) + load_jsonl(VAL)

    def test_no_join_question_appears_in_training(self, training, join_rows, bookstore_rows):
        train_q = {normalize_question(r["question"]) for r in training}
        for r in join_rows + bookstore_rows:
            assert normalize_question(r["question"]) not in train_q, r["question"]

    def test_no_join_gold_sql_appears_in_training(self, training, join_rows, bookstore_rows):
        train_sql = {normalize_sql(r["sql"]) for r in training}
        for r in join_rows + bookstore_rows:
            assert normalize_sql(r["sql"]) not in train_sql, r["sql"]

    def test_training_still_teaches_the_join_shapes(self, training):
        """De-leaking must not empty the join families it is meant to teach."""
        joins = [r for r in training if " join " in normalize_sql(r["sql"])]
        assert len(joins) > 50
        assert any("employees e join employees m" in normalize_sql(r["sql"]) for r in joins)
        assert any("group by departments.location" in normalize_sql(r["sql"]) for r in joins)
