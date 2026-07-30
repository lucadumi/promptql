"""Serve the fine-tuned text-to-SQL model behind a small HTTP API.

The last step of the loop: a model that only exists as a `results/*.json` number is
hard to believe in. This turns the adapter into something you can curl, and - because
the schema it was trained on has a real seeded database behind it (`src/db.py`) - the
API can *run* the SQL it generates and hand back actual rows. That end-to-end path,
question -> SQL -> rows, is the honest demonstration that the thing works.

Two deliberate constraints shape this file:

  * **No web framework.** It uses `http.server` from the standard library. The repo is
    dependency-light on purpose (CI installs neither torch nor transformers), and a
    text-to-SQL demo does not need FastAPI to prove anything.
  * **Importable without torch.** Every heavy import is deferred into `load_generator`,
    and the request handling is written against a plain `generate(question, schema) ->
    sql` callable. That is what lets `tests/test_serve.py` exercise the whole request
    path - routing, validation, error codes, SQL execution - in milliseconds with a
    stub generator, on a CI box that has no model.

Quantization (`--quantize`) applies PyTorch dynamic int8 quantization to the linear
layers. It is CPU-only (that is a PyTorch limitation, not a choice) and is the reason
`--device cpu` is the default here while the eval scripts default to MPS.

Run it (from the repo root, inside the venv):
    python -m src.serve                       # base + LoRA adapter, float32, CPU
    python -m src.serve --quantize            # dynamic int8 (smaller, CPU-only)
    python -m src.serve --no-adapter          # serve the base model for comparison

Then:
    curl localhost:8000/health
    curl -s localhost:8000/sql -d '{"question": "How many employees are there?"}'
    curl -s localhost:8000/sql -d '{"question": "...", "execute": false}'
"""
from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Callable, Dict, Optional, Tuple

# Make `src` importable whether run as `-m src.serve` or as a file path.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.db import SCHEMAS, build_db, run_sql  # noqa: E402
from src.metrics import extract_sql  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
DEFAULT_ADAPTER = REPO_ROOT / "adapters" / "lora-qwen2.5-0.5b-join"

# A generated query is answering a question, so anything that could modify the
# database is refused before it reaches SQLite. The connection is also rebuilt
# per request from the committed seed, so this is defence in depth rather than
# the only guard.
FORBIDDEN_SQL = ("insert", "update", "delete", "drop", "alter", "create",
                 "replace", "attach", "detach", "pragma")

# Statement keywords that mean "the model emitted SQL", whether or not we are
# willing to run it. Used only to tell "that isn't SQL at all" apart from "that
# is SQL I refuse to execute", so the caller gets an accurate reason.
SQL_STARTERS = ("select", "with") + FORBIDDEN_SQL

# question -> sql. The server only ever needs this much of a model.
Generator = Callable[[str, str], str]


def looks_like_sql(text: str) -> bool:
    """True if `text` begins a SQL statement of any kind.

    `extract_sql` returns its input unchanged when it finds no SQL keyword, so a
    chatty refusal ("I am afraid I cannot do that") comes back as a non-empty
    string. Eval scoring does not care - it just fails to execute and is marked
    wrong - but an API owes the caller an accurate reason, so "not SQL" and
    "SQL we will not run" are distinguished.
    """
    return text.strip().lower().startswith(SQL_STARTERS)


def is_safe_select(sql: str) -> bool:
    """True if `sql` is a single read-only SELECT statement.

    The model is asked for a SELECT and reliably produces one, but "reliably" is
    not "always", and this endpoint executes what it produces. Rejecting anything
    that is not a lone SELECT keeps a hallucinated `DROP TABLE` from ever reaching
    the database.
    """
    stripped = sql.strip().rstrip(";").strip().lower()
    if not stripped.startswith(("select", "with")):
        return False
    # A second statement smuggled in behind a semicolon.
    if ";" in sql.strip().rstrip(";"):
        return False
    return not any(f" {word} " in f" {stripped} " for word in FORBIDDEN_SQL)


def answer(
    question: str,
    generate: Generator,
    schema_name: str = "employees",
    execute: bool = True,
) -> Tuple[int, Dict[str, object]]:
    """Turn a question into an HTTP status and a JSON-ready response body.

    This is the whole application logic, deliberately kept free of both HTTP and
    torch so it can be tested directly. `generate` is any callable mapping
    (question, schema DDL) to raw model output.
    """
    if not question or not question.strip():
        return 400, {"error": "field 'question' must be a non-empty string"}
    if schema_name not in SCHEMAS:
        return 400, {"error": f"unknown schema {schema_name!r}",
                     "available": sorted(SCHEMAS)}

    schema = SCHEMAS[schema_name]
    raw = generate(question, schema.ddl)
    sql = extract_sql(raw)
    if not sql or not looks_like_sql(sql):
        return 422, {"error": "model did not produce a SQL query",
                     "question": question, "raw": raw}

    body: Dict[str, object] = {"question": question, "sql": sql, "schema": schema.name}
    if not execute:
        return 200, body

    if not is_safe_select(sql):
        body["error"] = "refusing to execute a non-SELECT statement"
        return 422, body

    conn = build_db(schema)
    try:
        rows = run_sql(conn, sql)
        body["columns"] = [d[0] for d in conn.execute(sql).description or []]
        body["rows"] = [list(r) for r in rows]
        body["row_count"] = len(rows)
        return 200, body
    except sqlite3.Error as exc:
        body["error"] = f"generated SQL failed to execute: {exc}"
        return 422, body
    finally:
        conn.close()


def load_generator(
    model_name: str = DEFAULT_MODEL,
    adapter: Optional[str] = None,
    device: str = "cpu",
    quantize: bool = False,
    max_new_tokens: int = 64,
) -> Generator:
    """Load the model once and return a (question, schema) -> raw SQL callable.

    Imports torch/transformers lazily so this module stays importable - and its
    request path stays testable - on a machine that has neither. Quantization is
    delegated to `src.eval_baseline` so the model served is byte-for-byte the one
    the reported numbers were measured on.
    """
    from src.eval_baseline import generate_sql, load_model_and_tokenizer

    model, tokenizer = load_model_and_tokenizer(model_name, device, adapter, quantize)

    def generate(question: str, schema_ddl: str) -> str:
        return generate_sql(model, tokenizer, question, device, max_new_tokens, schema_ddl)

    return generate


def make_handler(generate: Generator, default_schema: str = "employees"):
    """Build a request handler bound to a specific generator."""

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def _send(self, status: int, payload: Dict[str, object]) -> None:
            body = json.dumps(payload, indent=2, default=str).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:  # noqa: N802 - name fixed by BaseHTTPRequestHandler
            if self.path.rstrip("/") in ("/health", ""):
                self._send(200, {"status": "ok", "schemas": sorted(SCHEMAS)})
            else:
                self._send(404, {"error": f"no such endpoint: {self.path}",
                                 "endpoints": ["GET /health", "POST /sql"]})

        def do_POST(self) -> None:  # noqa: N802 - name fixed by BaseHTTPRequestHandler
            if self.path.rstrip("/") != "/sql":
                self._send(404, {"error": f"no such endpoint: {self.path}",
                                 "endpoints": ["GET /health", "POST /sql"]})
                return
            try:
                length = int(self.headers.get("Content-Length") or 0)
            except ValueError:
                self._send(400, {"error": "invalid Content-Length"})
                return
            try:
                payload = json.loads(self.rfile.read(length) or b"{}")
            except json.JSONDecodeError as exc:
                self._send(400, {"error": f"invalid JSON body: {exc}"})
                return
            if not isinstance(payload, dict):
                self._send(400, {"error": "body must be a JSON object"})
                return

            status, body = answer(
                question=payload.get("question", ""),
                generate=generate,
                schema_name=payload.get("schema", default_schema),
                execute=bool(payload.get("execute", True)),
            )
            self._send(status, body)

        def log_message(self, fmt: str, *args: object) -> None:
            sys.stderr.write("%s - %s\n" % (self.address_string(), fmt % args))

    return Handler


def main() -> int:
    parser = argparse.ArgumentParser(description="Serve the text-to-SQL model over HTTP.")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--adapter", default=str(DEFAULT_ADAPTER),
                        help="LoRA adapter directory (default: the shipped adapter)")
    parser.add_argument("--no-adapter", action="store_true",
                        help="serve the base model instead, for comparison")
    parser.add_argument("--quantize", action="store_true",
                        help="dynamic int8 quantization of the linear layers (CPU only)")
    parser.add_argument("--device", default="cpu", help="cpu | mps | cuda")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--schema", default="employees", choices=sorted(SCHEMAS),
                        help="default schema for requests that do not name one")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    args = parser.parse_args()

    adapter = None if args.no_adapter else args.adapter
    if adapter and not Path(adapter).exists():
        print(f"adapter not found: {adapter}\n"
              f"run `make data && make train`, or pass --no-adapter to serve the base model.",
              file=sys.stderr)
        return 1

    generate = load_generator(args.model, adapter, args.device, args.quantize,
                              args.max_new_tokens)

    server = ThreadingHTTPServer((args.host, args.port), make_handler(generate, args.schema))
    label = args.model + (f" + {Path(adapter).name}" if adapter else " (base)")
    print(f"Serving {label}{' [int8]' if args.quantize else ''} "
          f"on http://{args.host}:{args.port}  (schema: {args.schema})", flush=True)
    print("  GET  /health", flush=True)
    print("  POST /sql   {\"question\": \"...\", \"schema\": \"employees\", "
          "\"execute\": true}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nshutting down", flush=True)
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
