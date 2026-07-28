"""Unit tests for src/metrics.py -- the normalized-exact-match logic that every
before/after number in this project depends on. These are pure functions, so the
tests are fast and deterministic (no model, no torch).
"""
import pytest

from src.metrics import exact_match, extract_sql, normalize_sql


class TestExtractSql:
    def test_empty_returns_empty(self):
        assert extract_sql("") == ""
        assert extract_sql(None) == ""  # type: ignore[arg-type]

    def test_strips_markdown_fence(self):
        assert extract_sql("```sql\nSELECT 1 FROM t\n```") == "SELECT 1 FROM t"

    def test_strips_fence_without_language_tag(self):
        assert extract_sql("```\nSELECT 1 FROM t\n```") == "SELECT 1 FROM t"

    def test_cuts_leading_prose(self):
        assert extract_sql("Sure, here it is: SELECT id FROM users") == "SELECT id FROM users"

    def test_keeps_only_first_statement(self):
        assert extract_sql("SELECT 1 FROM a; SELECT 2 FROM b") == "SELECT 1 FROM a"

    def test_recognises_with_cte(self):
        text = "WITH x AS (SELECT 1) SELECT * FROM x"
        assert extract_sql(text) == text

    def test_keyword_match_is_case_insensitive(self):
        assert extract_sql("blah select id from t") == "select id from t"


class TestNormalizeSql:
    def test_collapses_whitespace(self):
        assert normalize_sql("SELECT   *\n  FROM   t") == "select * from t"

    def test_lowercases(self):
        assert normalize_sql("SELECT Id FROM Users") == "select id from users"

    def test_double_and_single_quotes_are_equal(self):
        assert normalize_sql('SELECT * FROM t WHERE c = "London"') == \
            normalize_sql("SELECT * FROM t WHERE c = 'London'")

    def test_strips_trailing_semicolon(self):
        assert normalize_sql("SELECT 1 FROM t;") == "select 1 from t"


class TestExactMatch:
    def test_true_despite_formatting_differences(self):
        pred = "```sql\nSELECT   *  FROM  t ;\n```"
        gold = "select * from t"
        assert exact_match(pred, gold) is True

    def test_quote_style_does_not_break_match(self):
        pred = 'SELECT name FROM employees WHERE department = "Sales"'
        gold = "SELECT name FROM employees WHERE department = 'Sales'"
        assert exact_match(pred, gold) is True

    def test_genuinely_different_queries_do_not_match(self):
        assert exact_match("SELECT a FROM t", "SELECT b FROM t") is False

    @pytest.mark.parametrize("pred,gold,expected", [
        ("SELECT COUNT(*) FROM departments", "select count(*) from departments", True),
        ("SELECT * FROM a", "SELECT * FROM b", False),
        ("", "SELECT 1", False),
    ])
    def test_table(self, pred, gold, expected):
        assert exact_match(pred, gold) is expected
