"""Unit tests for src/db.py -- the seed SQLite database and execution-accuracy
comparison. These are the honest upgrade over strict exact-match: a query that
returns the right rows is credited even when written differently. Pure SQLite +
Python, so the suite stays fast and needs no torch/model download.
"""
from src.db import (
    DEPARTMENTS_SEED,
    EMPLOYEES_SEED,
    ExecResult,
    build_db,
    execution_match,
    run_sql,
)
from src.metrics import exact_match


class TestSeedData:
    def test_row_counts(self):
        conn = build_db()
        assert run_sql(conn, "SELECT COUNT(*) FROM departments")[0][0] == 4
        assert run_sql(conn, "SELECT COUNT(*) FROM employees")[0][0] == 20

    def test_salaries_are_all_distinct(self):
        # Distinct salaries keep ORDER BY / LIMIT / MIN / MAX queries deterministic.
        salaries = [e[3] for e in EMPLOYEES_SEED]
        assert len(set(salaries)) == len(salaries)

    def test_engineering_has_more_than_ten_employees(self):
        # Drives the HAVING COUNT(*) > 10 eval query to a single department.
        conn = build_db()
        rows = run_sql(
            conn,
            "SELECT department FROM employees GROUP BY department HAVING COUNT(*) > 10",
        )
        assert rows == [("Engineering",)]

    def test_two_departments_in_new_york(self):
        conn = build_db()
        assert run_sql(
            conn, "SELECT COUNT(*) FROM departments WHERE location = 'New York'"
        )[0][0] == 2

    def test_salary_100000_filter_is_discriminating(self):
        # 11 of 20 employees earn > 100000 -> the filter isn't all-or-nothing.
        conn = build_db()
        assert run_sql(
            conn, "SELECT COUNT(*) FROM employees WHERE salary > 100000"
        )[0][0] == 11

    def test_unique_lowest_paid_employee(self):
        conn = build_db()
        rows = run_sql(conn, "SELECT name FROM employees ORDER BY salary ASC LIMIT 1")
        assert rows == [("Xavier",)]

    def test_seed_ids_are_unique(self):
        dept_ids = [d[0] for d in DEPARTMENTS_SEED]
        emp_ids = [e[0] for e in EMPLOYEES_SEED]
        assert len(set(dept_ids)) == len(dept_ids)
        assert len(set(emp_ids)) == len(emp_ids)


class TestExecutionMatchEquivalence:
    def test_identical_query_matches(self):
        sql = "SELECT COUNT(*) FROM employees"
        assert execution_match(sql, sql).match is True

    def test_count_star_vs_count_column_matches_by_execution(self):
        # The headline win: exact-match says no, execution says yes.
        pred = "SELECT COUNT(id) FROM employees"
        gold = "SELECT COUNT(*) FROM employees"
        assert exact_match(pred, gold) is False
        assert execution_match(pred, gold).match is True

    def test_gt_vs_gte_on_integer_boundary_matches(self):
        pred = "SELECT name FROM employees WHERE salary >= 100001"
        gold = "SELECT name FROM employees WHERE salary > 100000"
        assert exact_match(pred, gold) is False
        assert execution_match(pred, gold).match is True

    def test_markdown_fenced_prediction_is_executed(self):
        pred = "```sql\nSELECT COUNT(*) FROM employees\n```"
        gold = "SELECT COUNT(*) FROM employees"
        assert execution_match(pred, gold).match is True


class TestExecutionMatchOrdering:
    def test_unordered_gold_ignores_row_order(self):
        # Gold has no ORDER BY, so a differently-ordered prediction still matches.
        gold = "SELECT name FROM departments"
        pred = "SELECT name FROM departments ORDER BY name DESC"
        assert execution_match(pred, gold).match is True

    def test_ordered_gold_requires_matching_order(self):
        gold = "SELECT name FROM employees ORDER BY salary DESC"
        pred = "SELECT name FROM employees ORDER BY salary ASC"
        assert execution_match(pred, gold).match is False


class TestExecutionMatchFailures:
    def test_semantically_wrong_query_does_not_match(self):
        pred = "SELECT name FROM employees WHERE department = 'Marketing'"
        gold = "SELECT name FROM employees WHERE department = 'Sales'"
        assert execution_match(pred, gold).match is False

    def test_invalid_prediction_reports_error(self):
        res = execution_match("SELECT bogus FROM nowhere", "SELECT COUNT(*) FROM employees")
        assert res.match is False
        assert res.pred_error is not None

    def test_prediction_without_sql_reports_error(self):
        res = execution_match("I'm sorry, I can't answer that.", "SELECT COUNT(*) FROM employees")
        assert res.match is False
        assert res.pred_error is not None

    def test_broken_gold_is_surfaced_not_silently_passed(self):
        res = execution_match("SELECT 1", "SELECT * FROM missing_table")
        assert res.match is False
        assert res.gold_error is not None

    def test_returns_exec_result_instance(self):
        assert isinstance(execution_match("SELECT 1", "SELECT 1"), ExecResult)
