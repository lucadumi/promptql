"""Data helpers for the text-to-SQL baseline.

Keeps the fixed database schema, the prompt formatting, and JSONL loading in one
place so the eval script and (later) the training script use the *exact same*
prompt format. Prompt-format drift between train and eval is one of the pitfalls
called out in the README, so we centralise it here.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

# ---------------------------------------------------------------------------
# Fixed schema the eval questions are written against. The model is shown this
# schema in every prompt so the task is "read schema + question -> emit SQL".
# ---------------------------------------------------------------------------
SCHEMA_SQL = """\
CREATE TABLE departments (
    id       INTEGER PRIMARY KEY,
    name     TEXT,
    budget   INTEGER,
    location TEXT
);
CREATE TABLE employees (
    id         INTEGER PRIMARY KEY,
    name       TEXT,
    department TEXT,
    salary     INTEGER,
    hire_date  TEXT,      -- ISO date, e.g. '2021-06-01'
    manager_id INTEGER
);"""

SYSTEM_PROMPT = (
    "You are a precise text-to-SQL assistant. Given a database schema and a "
    "question, respond with a single valid SQLite query that answers it. "
    "Output ONLY the SQL query, with no explanation, comments, or markdown."
)


def build_user_prompt(question: str) -> str:
    """The user-turn text: schema + question."""
    return f"Schema:\n{SCHEMA_SQL}\n\nQuestion: {question}\nSQL:"


def build_messages(question: str) -> List[Dict[str, str]]:
    """Chat-format messages for instruct models (used with a chat template)."""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_user_prompt(question)},
    ]


def build_plain_prompt(question: str) -> str:
    """Fallback prompt for base (non-chat) models with no chat template."""
    return f"{SYSTEM_PROMPT}\n\n{build_user_prompt(question)}"


def load_jsonl(path: str | Path) -> List[Dict]:
    """Load a JSONL file into a list of dicts (blank lines ignored)."""
    path = Path(path)
    rows: List[Dict] = []
    with path.open("r", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:  # pragma: no cover - defensive
                raise ValueError(f"{path}:{line_no}: invalid JSON: {exc}") from exc
    return rows
