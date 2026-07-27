"""Metrics for the text-to-SQL baseline.

We keep this deliberately simple and honest: normalise both prediction and gold
SQL, then check for an exact string match. Exact-match is a strict lower bound on
quality (two different-but-equivalent queries count as wrong), which is fine for a
*baseline* -- it gives the fine-tuned model clear room to improve, and it never
overstates the base model. Execution-accuracy against a real SQLite DB is a good
Week-3 upgrade.
"""
from __future__ import annotations

import re

_FENCE_RE = re.compile(r"```(?:sql)?\s*(.*?)```", re.IGNORECASE | re.DOTALL)
_WS_RE = re.compile(r"\s+")


def extract_sql(text: str) -> str:
    """Pull the SQL query out of a raw model completion.

    Handles markdown code fences and leading chatter by taking the text from the
    first SELECT/WITH keyword onward, up to the first semicolon if present.
    """
    if not text:
        return ""

    # Prefer fenced code block content if the model wrapped its answer.
    fenced = _FENCE_RE.search(text)
    if fenced:
        text = fenced.group(1)

    # Cut leading prose: start at the first SQL keyword we recognise.
    match = re.search(r"\b(select|with|insert|update|delete)\b", text, re.IGNORECASE)
    if match:
        text = text[match.start():]

    # Keep only the first statement.
    if ";" in text:
        text = text.split(";", 1)[0]

    return text.strip()


def normalize_sql(sql: str) -> str:
    """Canonicalise a SQL string for exact-match comparison."""
    sql = extract_sql(sql)
    sql = sql.strip().rstrip(";").strip()
    sql = _WS_RE.sub(" ", sql)      # collapse all whitespace runs to one space
    sql = sql.replace('"', "'")     # treat single/double string quotes alike
    return sql.lower()


def exact_match(pred: str, gold: str) -> bool:
    """True if prediction matches gold after normalisation."""
    return normalize_sql(pred) == normalize_sql(gold)
