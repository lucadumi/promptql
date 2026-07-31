"""Execute-and-repair: let the model see its own SQLite error and try again.

Every result in this repo so far measures a single greedy generation. But the
deployment path already has something the eval loop was throwing away: the API in
``src/serve.py`` *runs* the SQL it produces, so when a query is invalid, SQLite
says exactly what is wrong with it. Feeding that message back is free information
at inference time, and it needs no retraining.

The headroom is worth stating before the mechanism, because it bounds what this
can possibly achieve. Of the shipped adapter's 22 failures across all seven eval
sets, **15 raise a SQLite error** and 7 run fine but return the wrong rows. This
loop can only ever address the first group: without the gold query there is no way
to know that a query which executed cleanly answered the wrong question. So the
ceiling is "fix some of 15", not "fix 22", and any honest report of it has to show
the two groups separately.

Three properties make this legitimate rather than a way of peeking at the answers:

  * **It never sees the gold.** The only signal is whether SQLite accepted the
    query - information any production caller has, and which says nothing about
    whether the answer is right.
  * **It cannot make a score worse.** A repair is attempted only for a query that
    already failed to execute, and a query that fails to execute is already graded
    wrong on both metrics. So the loop is monotone: it fixes things or does
    nothing. That is a property of the design, and `tests/test_repair.py` pins it.
  * **It does not retry on an empty result set.** Returning no rows is a perfectly
    good answer to some questions, and treating "0 rows" as failure would be a
    heuristic that quietly punishes correct queries. Only an outright SQLite error
    counts as a failure worth retrying.

The retry prompt deliberately stays *single-turn*. The obvious alternative is a
chat continuation (assistant turn with the bad SQL, then a user turn with the
error), but the fine-tune only ever saw one-shot prompts, and this project has
already learned twice over that a small model punishes out-of-distribution input
harder than it rewards extra context. So the error is appended inside the same
question field the model already knows, and the prompt format is otherwise
byte-identical to training.

Pure stdlib: no torch anywhere in this module, so `tests/test_repair.py` exercises
the whole loop with a stub generator in milliseconds.
"""
from __future__ import annotations

import re
import sqlite3
from dataclasses import dataclass, field
from typing import Callable, List, Optional

from src.db import Schema, build_db
from src.metrics import extract_sql

# The retry question: the original text plus a directive derived from the error.
#
# What is *absent* matters more than what is present. The first version of this
# appended the failed query and the raw SQLite message, and the fine-tuned model
# ignored it completely -- the retry was byte-identical to the first attempt. Two
# changes fixed that, both found by probing the model directly:
#
#   * **Do not show the model its own failed SQL.** Under greedy decoding, putting
#     the previous answer in the prompt is a strong prior to reproduce it. The
#     failed query is still recorded in RepairResult for auditing; it is just kept
#     out of the prompt.
#   * **Translate the error into an instruction, and make it corrective rather
#     than prohibitive.** "no such column: x.y" is a description of a fact, not
#     something to act on. But "Do NOT use the column x.y" over-corrects: told that,
#     the model avoided the *whole table* and invented a new one for the projection.
#     Pointing at the fix ("that column does not exist; use the correct one from the
#     schema") produced exactly the gold query on the same question.
REPAIR_TEMPLATE = "{question}\n{directive}"

_NO_SUCH_RE = re.compile(r"no such (column|table|function):\s*(.+?)\s*$",
                         re.IGNORECASE | re.MULTILINE)

# Raw model output, and the callable that turns a question into it.
Generator = Callable[[str], str]
# Returns None if the SQL runs, or the error message if it does not.
Validator = Callable[[str], Optional[str]]


def error_hint(error: str) -> str:
    """Turn a SQLite error message into an instruction the model can act on.

    Deliberately generic: it parses the *shape* of SQLite's message, so nothing
    here is tuned to a particular schema, column or eval set.
    """
    match = _NO_SUCH_RE.search(error or "")
    if match:
        kind, name = match.group(1).lower(), match.group(2).strip()
        if kind == "column":
            return (f"The column {name} does not exist. Look at the schema above "
                    "and use the correct column name from the right table.")
        if kind == "table":
            return (f"There is no table called {name}. Look at the schema above "
                    "and use the correct table.")
        return (f"SQLite has no function {name}. "
                "Use an expression that SQLite supports instead.")
    return (f"The previous attempt failed with: {error}. "
            "Write a valid SQLite query using only the schema above.")


@dataclass
class RepairResult:
    """What the loop settled on, plus enough history to audit it."""

    raw: str
    sql: str
    attempts: int = 1
    errors: List[str] = field(default_factory=list)

    @property
    def repaired(self) -> bool:
        """True if an attempt failed and a later one produced runnable SQL."""
        return bool(self.errors) and len(self.errors) < self.attempts


def build_repair_question(question: str, error: str) -> str:
    """The retry question: the original text plus a directive built from the error.

    The failed SQL is deliberately *not* included; see REPAIR_TEMPLATE.
    """
    return REPAIR_TEMPLATE.format(question=question.strip(),
                                  directive=error_hint(error))


def sqlite_validator(schema: Schema) -> Validator:
    """A validator that runs SQL against a fresh seeded database.

    The database is rebuilt per call from the committed seed, so a query can
    never leave state behind for the next one.
    """

    def validate(sql: str) -> Optional[str]:
        if not sql:
            return "no SQL found in the model output"
        conn = build_db(schema)
        try:
            conn.execute(sql).fetchall()
            return None
        except sqlite3.Error as exc:
            return str(exc)
        finally:
            conn.close()

    return validate


def generate_with_repair(
    generate: Generator,
    question: str,
    validate: Validator,
    max_attempts: int = 2,
) -> RepairResult:
    """Generate SQL, and re-ask with the error if SQLite rejects it.

    ``generate`` maps a question to raw model output; ``validate`` returns None
    when the extracted SQL runs and an error message when it does not.
    ``max_attempts`` counts the first try, so the default of 2 means "one repair".

    Returns the first attempt that executes. If every attempt fails, the *last*
    one is returned - all of them are graded wrong either way, and the last is the
    most informed, so it is the more useful thing to show a caller. In that case
    ``errors`` holds one entry per attempt, so ``repaired`` stays False; the final
    attempt is validated too rather than assumed good.
    """
    if max_attempts < 1:
        raise ValueError("max_attempts must be at least 1")

    raw = generate(question)
    result = RepairResult(raw=raw, sql=extract_sql(raw))

    for attempt in range(1, max_attempts + 1):
        error = validate(result.sql)
        if error is None:
            return result
        result.errors.append(error)
        if attempt == max_attempts:
            return result           # out of budget, last attempt kept as-is
        retry_question = build_repair_question(question, error)
        raw = generate(retry_question)
        result.raw = raw
        result.sql = extract_sql(raw)
        result.attempts += 1

    return result
