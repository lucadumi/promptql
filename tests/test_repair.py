"""Unit tests for the execute-and-repair loop.

The loop re-asks the model when SQLite rejects its query, feeding back the error.
Two things make it worth testing carefully rather than trusting:

  * it is only defensible while it never sees the gold answer, and
  * it is only safe to enable by default while it *cannot* make a score worse.

The second is a design property, not a hope: a repair is attempted only for a
query that already failed to execute, and such a query is already graded wrong on
both metrics. These tests pin that, plus the loop's stopping conditions.

Pure stdlib with a stub generator, so the whole file runs in milliseconds without
torch or a model.
"""
import pytest

from src.db import EMPLOYEES_SCHEMA, build_db, run_sql
from src.repair import (
    build_repair_question,
    generate_with_repair,
    sqlite_validator,
)


class ScriptedGenerator:
    """Returns a fixed list of completions, recording the questions it was asked."""

    def __init__(self, *completions: str):
        self.completions = list(completions)
        self.questions = []

    def __call__(self, question: str) -> str:
        self.questions.append(question)
        idx = min(len(self.questions) - 1, len(self.completions) - 1)
        return self.completions[idx]


@pytest.fixture(scope="module")
def validate():
    return sqlite_validator(EMPLOYEES_SCHEMA)


def validate_module_level():
    """Same validator, for tests that are not written as fixtures."""
    return sqlite_validator(EMPLOYEES_SCHEMA)


class TestValidator:
    def test_accepts_runnable_sql(self, validate):
        assert validate("SELECT name FROM employees") is None

    def test_reports_the_sqlite_error(self, validate):
        error = validate("SELECT name FROM teams")
        assert error and "teams" in error

    def test_reports_empty_input(self, validate):
        assert validate("") is not None

    def test_an_empty_result_set_is_not_an_error(self, validate):
        """Returning no rows is a legitimate answer to some questions. Treating it
        as failure would make the loop retry correct queries."""
        assert run_sql(build_db(EMPLOYEES_SCHEMA),
                       "SELECT name FROM employees WHERE salary > 999999") == []
        assert validate("SELECT name FROM employees WHERE salary > 999999") is None


class TestLoop:
    def test_a_working_query_is_returned_without_a_retry(self, validate):
        gen = ScriptedGenerator("SELECT name FROM employees")
        result = generate_with_repair(gen, "List everyone.", validate, max_attempts=2)
        assert result.attempts == 1
        assert result.errors == []
        assert result.repaired is False
        assert len(gen.questions) == 1

    def test_a_broken_query_is_retried_and_fixed(self, validate):
        gen = ScriptedGenerator("SELECT name FROM teams",
                                "SELECT department FROM employees")
        result = generate_with_repair(gen, "List the teams.", validate, max_attempts=2)
        assert result.attempts == 2
        assert result.repaired is True
        assert result.sql == "SELECT department FROM employees"
        assert len(result.errors) == 1 and "teams" in result.errors[0]

    def test_repair_is_off_when_max_attempts_is_one(self, validate):
        gen = ScriptedGenerator("SELECT name FROM teams", "SELECT name FROM employees")
        result = generate_with_repair(gen, "List the teams.", validate, max_attempts=1)
        assert result.attempts == 1
        assert result.sql == "SELECT name FROM teams"
        assert len(gen.questions) == 1

    def test_the_last_attempt_is_returned_when_every_try_fails(self, validate):
        gen = ScriptedGenerator("SELECT name FROM teams", "SELECT name FROM squads")
        result = generate_with_repair(gen, "List the teams.", validate, max_attempts=2)
        assert result.attempts == 2
        assert result.repaired is False
        assert result.sql == "SELECT name FROM squads"

    def test_it_keeps_trying_up_to_the_budget(self, validate):
        gen = ScriptedGenerator("SELECT * FROM a", "SELECT * FROM b",
                                "SELECT name FROM employees")
        result = generate_with_repair(gen, "q", validate, max_attempts=3)
        assert result.attempts == 3
        assert result.repaired is True
        assert len(result.errors) == 2

    def test_zero_attempts_is_rejected(self, validate):
        with pytest.raises(ValueError):
            generate_with_repair(ScriptedGenerator("x"), "q", validate, max_attempts=0)


class TestRetryPrompt:
    def test_it_turns_a_missing_column_into_an_instruction(self):
        prompt = build_repair_question("How many teams?",
                                       "no such column: employees.team")
        assert "How many teams?" in prompt
        assert "column employees.team does not exist" in prompt

    def test_it_turns_a_missing_table_into_an_instruction(self):
        assert "no table called teams" in build_repair_question(
            "q", "no such table: teams")

    def test_it_turns_a_missing_function_into_an_instruction(self):
        assert "SQLite has no function YEAR" in build_repair_question(
            "q", "no such function: YEAR")

    def test_an_unrecognised_error_falls_back_to_quoting_it(self):
        msg = "aggregate functions are not allowed in the GROUP BY clause"
        assert msg in build_repair_question("q", msg)

    def test_the_failed_sql_is_not_shown_to_the_model(self):
        """Under greedy decoding, putting the previous answer in the prompt is a
        strong prior to reproduce it: the first version of this loop did exactly
        that and every retry came back byte-identical. The failed query stays in
        RepairResult for auditing, but out of the prompt."""
        gen = ScriptedGenerator("SELECT name FROM teams",
                                "SELECT department FROM employees")
        generate_with_repair(gen, "Which teams exist?", validate_module_level(),
                             max_attempts=2)
        assert "SELECT name FROM teams" not in gen.questions[1]

    def test_the_retry_asks_the_original_question_again(self):
        """The model has to answer the *question*, not merely produce runnable SQL.
        Dropping the original text would let it 'fix' the error with anything that
        parses."""
        gen = ScriptedGenerator("SELECT name FROM teams",
                                "SELECT department FROM employees")
        generate_with_repair(gen, "Which teams exist?", validate_module_level(),
                             max_attempts=2)
        assert gen.questions[0] == "Which teams exist?"
        assert "Which teams exist?" in gen.questions[1]

    def test_it_survives_a_completion_with_no_sql_in_it(self, validate):
        gen = ScriptedGenerator("I'm not sure how to answer that.",
                                "SELECT name FROM employees")
        result = generate_with_repair(gen, "q", validate, max_attempts=2)
        assert result.repaired is True
        assert result.sql == "SELECT name FROM employees"
        assert result.errors, "the failed attempt should still be recorded"


class TestCannotMakeThingsWorse:
    """The property that lets this be enabled by default."""

    def test_a_query_that_runs_is_never_replaced(self, validate):
        """Even a *wrong* query is left alone if it executes: without the gold
        there is no way to tell it is wrong, and swapping it for another guess
        could turn a correct answer into an incorrect one."""
        wrong_but_valid = "SELECT COUNT(*) FROM employees"
        gen = ScriptedGenerator(wrong_but_valid, "SELECT name FROM employees")
        result = generate_with_repair(gen, "List everyone.", validate, max_attempts=3)
        assert result.sql == wrong_but_valid
        assert result.attempts == 1

    def test_repair_only_ever_replaces_a_query_that_failed_to_execute(self, validate):
        """Restated as an invariant over the recorded history: every attempt the
        loop abandoned had produced an error."""
        gen = ScriptedGenerator("SELECT * FROM nope", "SELECT * FROM nah",
                                "SELECT name FROM employees")
        result = generate_with_repair(gen, "q", validate, max_attempts=3)
        assert len(result.errors) == result.attempts - 1
        assert all(e for e in result.errors)
