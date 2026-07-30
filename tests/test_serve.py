"""Unit tests for the HTTP serving layer (`src/serve.py`).

The point of these tests is that they run **without torch**. `src/serve.py` defers
every heavy import into `load_generator` and expresses its logic against a plain
`generate(question, schema) -> sql` callable, so the entire request path - routing,
validation, status codes, SQL safety, execution against the seeded DB - can be
exercised in milliseconds with a stub generator on a CI box that has no model.

That is not a testing trick, it is the design constraint that makes the serving layer
reviewable: everything except "what does the model say" is deterministic, and it is
all covered here.
"""
import json
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

import pytest

from src.serve import answer, is_safe_select, make_handler


def const_generator(sql: str):
    """A stub model: whatever SQL you hand it, that is what it 'generates'."""
    def generate(question: str, schema_ddl: str) -> str:
        return sql
    return generate


def echo_schema_generator():
    """A stub that reveals which schema DDL it was handed."""
    def generate(question: str, schema_ddl: str) -> str:
        table = "books" if "books" in schema_ddl else "employees"
        return f"SELECT COUNT(*) FROM {table}"
    return generate


class TestSafetyFilter:
    @pytest.mark.parametrize("sql", [
        "SELECT * FROM employees",
        "select name from employees where salary > 100",
        "SELECT COUNT(*) FROM employees;",
        "WITH x AS (SELECT 1) SELECT * FROM x",
    ])
    def test_accepts_read_only_selects(self, sql):
        assert is_safe_select(sql) is True

    @pytest.mark.parametrize("sql", [
        "DROP TABLE employees",
        "DELETE FROM employees",
        "INSERT INTO employees VALUES (1)",
        "UPDATE employees SET salary = 0",
        "PRAGMA table_info(employees)",
        "ATTACH DATABASE 'x.db' AS x",
        "SELECT 1; DROP TABLE employees",       # statement smuggled behind a ;
        "SELECT 1; DROP TABLE employees;",      # ... and with a trailing ;
        "",
    ])
    def test_rejects_anything_that_could_write(self, sql):
        assert is_safe_select(sql) is False


class TestAnswerValidation:
    def test_empty_question_is_a_400(self):
        status, body = answer("", const_generator("SELECT 1"))
        assert status == 400 and "question" in body["error"]

    def test_whitespace_question_is_a_400(self):
        status, body = answer("   ", const_generator("SELECT 1"))
        assert status == 400

    def test_unknown_schema_is_a_400_and_lists_the_valid_ones(self):
        status, body = answer("q", const_generator("SELECT 1"), schema_name="nope")
        assert status == 400
        assert "employees" in body["available"] and "bookstore" in body["available"]

    def test_model_output_without_sql_is_a_422(self):
        status, body = answer("q", const_generator("I am afraid I cannot do that"))
        assert status == 422
        assert "did not produce" in body["error"]


class TestAnswerHappyPath:
    def test_returns_sql_rows_and_columns(self):
        status, body = answer("How many employees are there?",
                              const_generator("SELECT COUNT(*) FROM employees"))
        assert status == 200
        assert body["sql"] == "SELECT COUNT(*) FROM employees"
        assert body["rows"] == [[20]]
        assert body["row_count"] == 1
        assert body["columns"] == ["COUNT(*)"]
        assert body["schema"] == "employees"

    def test_execute_false_skips_the_database_entirely(self):
        # A query that would fail on the DB still returns 200 when not executed,
        # which is what makes `execute: false` useful for inspecting raw output.
        status, body = answer("q", const_generator("SELECT nope FROM employees"),
                              execute=False)
        assert status == 200
        assert "rows" not in body and "error" not in body

    def test_markdown_fenced_output_is_unwrapped(self):
        fenced = "```sql\nSELECT COUNT(*) FROM employees\n```"
        status, body = answer("q", const_generator(fenced))
        assert status == 200 and body["sql"] == "SELECT COUNT(*) FROM employees"

    def test_the_named_schema_reaches_the_generator_and_the_database(self):
        status, body = answer("how many?", echo_schema_generator(),
                              schema_name="bookstore")
        assert status == 200
        assert body["sql"] == "SELECT COUNT(*) FROM books"
        assert body["schema"] == "bookstore"
        assert body["rows"] == [[20]]


class TestAnswerFailureModes:
    def test_broken_sql_is_reported_not_raised(self):
        status, body = answer("q", const_generator("SELECT nope FROM employees"))
        assert status == 422
        assert "failed to execute" in body["error"]
        assert body["sql"] == "SELECT nope FROM employees"   # still shown, for debugging

    def test_a_destructive_query_is_refused_before_the_database(self):
        status, body = answer("drop everything", const_generator("DROP TABLE employees"))
        assert status == 422
        assert "non-SELECT" in body["error"]
        assert "rows" not in body

    def test_refusing_a_write_leaves_the_seed_database_intact(self):
        """The refusal must be real, not cosmetic."""
        answer("q", const_generator("DELETE FROM employees"))
        status, body = answer("q", const_generator("SELECT COUNT(*) FROM employees"))
        assert body["rows"] == [[20]]


@pytest.fixture()
def server():
    """A real HTTP server on an ephemeral port, backed by a stub generator."""
    handler = make_handler(const_generator("SELECT COUNT(*) FROM employees"))
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}"
    httpd.shutdown()
    httpd.server_close()
    thread.join(timeout=5)


def post(url: str, payload, raw: bytes = None):
    data = raw if raw is not None else json.dumps(payload).encode()
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return resp.status, json.loads(resp.read())
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read())


class TestHttpEndpoints:
    def test_health_reports_the_available_schemas(self, server):
        with urllib.request.urlopen(f"{server}/health", timeout=10) as resp:
            assert resp.status == 200
            body = json.loads(resp.read())
        assert body["status"] == "ok"
        assert sorted(body["schemas"]) == ["bookstore", "employees"]

    def test_post_sql_returns_generated_sql_and_rows(self, server):
        status, body = post(f"{server}/sql", {"question": "How many employees?"})
        assert status == 200
        assert body["sql"] == "SELECT COUNT(*) FROM employees"
        assert body["rows"] == [[20]]

    def test_unknown_path_is_a_404_that_lists_the_endpoints(self, server):
        status, body = post(f"{server}/nope", {"question": "x"})
        assert status == 404
        assert "POST /sql" in body["endpoints"]

    def test_unknown_get_path_is_a_404(self, server):
        try:
            urllib.request.urlopen(f"{server}/nope", timeout=10)
            raise AssertionError("expected a 404")
        except urllib.error.HTTPError as exc:
            assert exc.code == 404

    def test_malformed_json_is_a_400_not_a_crash(self, server):
        status, body = post(f"{server}/sql", None, raw=b"{not json")
        assert status == 400 and "invalid JSON" in body["error"]

    def test_non_object_body_is_a_400(self, server):
        status, body = post(f"{server}/sql", ["a", "list"])
        assert status == 400 and "JSON object" in body["error"]

    def test_missing_question_is_a_400(self, server):
        status, body = post(f"{server}/sql", {})
        assert status == 400

    def test_server_survives_a_bad_request_and_serves_the_next_one(self, server):
        post(f"{server}/sql", None, raw=b"{not json")
        status, _ = post(f"{server}/sql", {"question": "How many employees?"})
        assert status == 200
