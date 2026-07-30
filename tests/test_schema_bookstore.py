"""Unit tests for the second (bookstore) schema used by the cross-schema eval.

data/eval/text2sql_eval_bookstore.jsonl re-tests the same SQL constructs as the
in-template eval, but against a completely different schema (publishers + books,
all-new names) that the model never trained on. Scoring it measures whether the
fine-tune transfers the task or only memorised the employees column names. These
tests guard the pieces that make that measurement trustworthy:

  * the bookstore seed keeps every eval query discriminating (it mirrors the
    employees invariants -- distinct prices, one genre with > 10 books, etc.), and
  * schema selection is real: a bookstore gold only runs on the bookstore DB, and
    execution_match builds the database named by its schema argument.

Pure data + SQLite, so the suite stays fast (no torch / model download).
"""
import sqlite3
from pathlib import Path

import pytest

from src.data_utils import BOOKSTORE_SCHEMA_SQL, build_user_prompt, load_jsonl
from src.db import (
    BOOKSTORE_SCHEMA,
    EMPLOYEES_SCHEMA,
    SCHEMAS,
    build_db,
    execution_match,
    run_sql,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
BOOKSTORE = REPO_ROOT / "data" / "eval" / "text2sql_eval_bookstore.jsonl"
ORIGINAL = REPO_ROOT / "data" / "eval" / "text2sql_eval.jsonl"


class TestBookstoreShape:
    def test_has_twenty_well_formed_rows(self):
        rows = load_jsonl(BOOKSTORE)
        assert len(rows) == 20
        for r in rows:
            assert set(r) >= {"id", "question", "sql"}
            assert r["question"].strip()
            assert r["sql"].strip()

    def test_ids_are_unique(self):
        ids = [r["id"] for r in load_jsonl(BOOKSTORE)]
        assert len(set(ids)) == len(ids)


class TestExecutableOnSeedDb:
    def test_every_gold_runs_and_returns_rows(self):
        # Non-empty results keep execution accuracy discriminating on the seed DB.
        conn = build_db(BOOKSTORE_SCHEMA)
        for r in load_jsonl(BOOKSTORE):
            assert run_sql(conn, r["sql"]), f"empty/failed gold for id {r['id']}: {r['sql']}"


class TestConstructCoverage:
    def test_covers_the_same_sql_constructs_as_the_in_template_eval(self):
        # The point of a second schema is to re-test the SAME skills, so every
        # construct the original eval exercises must appear here too.
        golds = " ".join(r["sql"].upper() for r in load_jsonl(BOOKSTORE))
        for kw in (
            "COUNT(", "AVG(", "SUM(", "MAX(", "DISTINCT",
            "WHERE", "ORDER BY", "LIMIT", "GROUP BY", "HAVING",
        ):
            assert kw in golds, f"bookstore eval is missing construct: {kw}"


class TestSeedInvariants:
    def test_prices_are_all_distinct(self):
        conn = build_db(BOOKSTORE_SCHEMA)
        total = run_sql(conn, "SELECT COUNT(*) FROM books")[0][0]
        distinct = run_sql(conn, "SELECT COUNT(DISTINCT price) FROM books")[0][0]
        assert total == distinct == 20

    def test_one_genre_exceeds_ten_books(self):
        conn = build_db(BOOKSTORE_SCHEMA)
        rows = run_sql(conn, "SELECT genre FROM books GROUP BY genre HAVING COUNT(*) > 10")
        assert rows == [("Fiction",)]

    def test_publisher_filters_are_discriminating(self):
        conn = build_db(BOOKSTORE_SCHEMA)
        assert run_sql(conn, "SELECT COUNT(*) FROM publishers WHERE city = 'New York'")[0][0] == 2
        assert run_sql(conn, "SELECT COUNT(*) FROM publishers WHERE revenue > 500000")[0][0] == 2


class TestSchemaSelection:
    def test_registry_exposes_both_schemas(self):
        assert set(SCHEMAS) == {"employees", "bookstore"}
        assert SCHEMAS["bookstore"] is BOOKSTORE_SCHEMA
        assert SCHEMAS["employees"] is EMPLOYEES_SCHEMA

    def test_bookstore_gold_needs_the_bookstore_db(self):
        # A bookstore query only runs where the books table exists, proving the
        # two schemas are genuinely distinct (not a shared table set).
        conn = build_db(EMPLOYEES_SCHEMA)
        with pytest.raises(sqlite3.Error):
            run_sql(conn, "SELECT title FROM books")

    def test_execution_match_builds_the_named_schema(self):
        gold = "SELECT title FROM books ORDER BY price ASC LIMIT 1"
        # Right schema: the gold matches itself.
        assert execution_match(gold, gold, BOOKSTORE_SCHEMA).match is True
        # Default (employees) schema: the gold cannot run, so it is a non-match.
        res = execution_match(gold, gold)
        assert res.match is False
        assert res.gold_error is not None


class TestPromptCarriesSchema:
    def test_user_prompt_embeds_the_selected_schema(self):
        prompt = build_user_prompt("How many books are there?", BOOKSTORE_SCHEMA_SQL)
        assert "books" in prompt and "publishers" in prompt
        assert "employees" not in prompt


class TestSchemaArgIsBackwardCompatible:
    """Adding the schema parameter must not break existing single-schema callers.

    src/train_lora.py reuses eval_baseline.generate_sql for its end-of-run sample
    generations and calls it without a schema, so schema_sql has to stay optional.
    Making it required once broke training at the very last line, after the
    adapter had already been saved. Signature-only checks, so no torch needed.
    """

    def test_generate_sql_can_be_called_without_a_schema(self):
        import inspect

        from src.eval_baseline import generate_sql

        sig = inspect.signature(generate_sql)
        assert sig.parameters["schema_sql"].default is not inspect.Parameter.empty
        # The exact call train_lora.sample_predictions makes must bind cleanly.
        sig.bind(object(), object(), "question", "cpu", 64)

    def test_generate_sql_default_schema_is_the_employees_schema(self):
        import inspect

        from src.data_utils import SCHEMA_SQL
        from src.eval_baseline import generate_sql

        default = inspect.signature(generate_sql).parameters["schema_sql"].default
        assert default == SCHEMA_SQL

    def test_prompt_builders_and_db_helpers_keep_working_with_one_argument(self):
        # Same contract for the other functions the schema refactor touched.
        assert "employees" in build_user_prompt("How many employees are there?")
        conn = build_db()
        try:
            assert run_sql(conn, "SELECT COUNT(*) FROM employees")
        finally:
            conn.close()
