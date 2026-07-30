"""Unit tests for the held-back ("blind") eval set.

Every other set in `data/eval` has fed at least one data-curation decision: the
balancing cap, the ambiguity fix, the literal-shape balance and the starved-pattern
fix were each diagnosed by reading eval failures. That makes them development
signal rather than an unbiased estimate. `data/eval/text2sql_eval_blind.jsonl` is
the answer to that: 24 fresh intents written *after* the shipped adapter was
trained and frozen, scored once, and never used to steer the training data.

A blind set is a process guarantee as much as a data file, so these tests pin the
properties that keep it meaningful:

  * it is genuinely fresh - no question or gold overlaps another eval set, and
    nothing collides with what the generator can produce (so de-leaking it costs
    the training set nothing, and the shipped adapter never saw these targets),
  * every gold runs on the seed DB and discriminates, and
  * it stays OUT of the routine `eval-all` regression loop, which is the one
    mechanical thing that would silently turn it back into dev signal.

Pure data + SQLite, so the suite stays fast (no torch / model download).
"""
import re
from pathlib import Path

import pytest

from src.build_dataset import generate_candidates, normalize_question
from src.data_utils import load_jsonl
from src.db import EMPLOYEES_SCHEMA, build_db, run_sql
from src.metrics import normalize_sql

REPO_ROOT = Path(__file__).resolve().parents[1]
EVAL_DIR = REPO_ROOT / "data" / "eval"
BLIND = EVAL_DIR / "text2sql_eval_blind.jsonl"
TRAIN = REPO_ROOT / "data" / "train" / "text2sql_train.jsonl"
VAL = REPO_ROOT / "data" / "train" / "text2sql_val.jsonl"
MAKEFILE = REPO_ROOT / "Makefile"


@pytest.fixture(scope="module")
def blind():
    return load_jsonl(BLIND)


@pytest.fixture(scope="module")
def other_eval_rows():
    rows = []
    for path in sorted(EVAL_DIR.glob("*.jsonl")):
        if path.name != BLIND.name:
            rows.extend(load_jsonl(path))
    return rows


class TestShape:
    def test_has_well_formed_rows(self, blind):
        assert len(blind) == 24
        for r in blind:
            assert set(r) >= {"id", "question", "sql"}
            assert r["question"].strip()
            assert r["sql"].strip()

    def test_ids_are_unique(self, blind):
        ids = [r["id"] for r in blind]
        assert len(set(ids)) == len(ids)

    def test_questions_are_unique(self, blind):
        qs = [normalize_question(r["question"]) for r in blind]
        assert len(set(qs)) == len(qs)


class TestExecutableOnSeedDb:
    def test_every_gold_runs_and_returns_rows(self, blind):
        # Non-empty results keep execution accuracy discriminating on the seed DB.
        conn = build_db(EMPLOYEES_SCHEMA)
        try:
            for r in blind:
                assert run_sql(conn, r["sql"]), f"empty/failed gold {r['id']}: {r['sql']}"
        finally:
            conn.close()

    def test_no_gold_returns_the_entire_employees_table(self, blind):
        """A gold matching all 20 rows barely discriminates on a single table."""
        conn = build_db(EMPLOYEES_SCHEMA)
        try:
            for r in blind:
                if " join " in normalize_sql(r["sql"]):
                    continue  # a join over every employee is legitimately 20 rows
                assert len(run_sql(conn, r["sql"])) < 20, f"gold {r['id']} is not selective"
        finally:
            conn.close()


class TestGenuinelyUnseen:
    """Blind means unseen by the model AND unseen by the data generator."""

    def test_no_question_appears_in_another_eval_set(self, blind, other_eval_rows):
        seen = {normalize_question(r["question"]) for r in other_eval_rows}
        assert [r["id"] for r in blind if normalize_question(r["question"]) in seen] == []

    def test_no_gold_appears_in_another_eval_set(self, blind, other_eval_rows):
        seen = {normalize_sql(r["sql"]) for r in other_eval_rows}
        assert [r["id"] for r in blind if normalize_sql(r["sql"]) in seen] == []

    def test_nothing_collides_with_the_shipped_training_data(self, blind):
        training = load_jsonl(TRAIN) + load_jsonl(VAL)
        tq = {normalize_question(r["question"]) for r in training}
        ts = {normalize_sql(r["sql"]) for r in training}
        assert [r["id"] for r in blind if normalize_question(r["question"]) in tq] == []
        assert [r["id"] for r in blind if normalize_sql(r["sql"]) in ts] == []

    def test_nothing_the_generator_can_produce_collides(self, blind):
        """The strong form: not just absent from the current split, but
        unreachable by the generator at all. This is what lets the blind set be
        added without changing a single training example, so the number it
        produces describes the adapter that was already shipped."""
        candidates = generate_candidates()
        cq = {normalize_question(q) for _c, q, _s in candidates}
        cs = {normalize_sql(s) for _c, _q, s in candidates}
        assert [r["id"] for r in blind if normalize_question(r["question"]) in cq] == []
        assert [r["id"] for r in blind if normalize_sql(r["sql"]) in cs] == []


class TestCoverage:
    def test_covers_the_constructs_the_dev_sets_test(self, blind):
        golds = " ".join(r["sql"].upper() for r in blind)
        for kw in ("COUNT(", "AVG(", "MAX(", "MIN(", "DISTINCT", "WHERE",
                   "ORDER BY", "LIMIT", "GROUP BY", "HAVING", "JOIN"):
            assert kw in golds, f"blind set is missing construct: {kw}"

    def test_also_reaches_beyond_what_training_teaches(self, blind):
        """A blind set that only re-tested trained shapes would measure very
        little. These constructs appear in no training template, so the set can
        genuinely surprise us."""
        golds = " ".join(r["sql"].upper() for r in blind)
        for kw in ("BETWEEN", "LIKE", "IS NULL", "!=", "SELECT DISTINCT"):
            assert kw in golds, f"blind set is missing untrained construct: {kw}"

    def test_includes_a_correlated_subquery(self, blind):
        nested = [r for r in blind if r["sql"].upper().count("SELECT") > 1]
        assert nested, "expected at least one subquery gold"

    def test_includes_a_self_join(self, blind):
        assert any("employees e join employees m" in normalize_sql(r["sql"]) for r in blind)


class TestHeldBackFromTheRegressionLoop:
    """The process guarantee, enforced mechanically rather than by discipline."""

    def _target_body(self, name: str) -> str:
        text = MAKEFILE.read_text()
        match = re.search(rf"^{name}:\n((?:\t.*\n|\n(?=\t))*)", text, re.M)
        assert match, f"no {name} target in the Makefile"
        return match.group(1)

    def test_eval_all_does_not_score_the_blind_set(self):
        assert "blind" not in self._target_body("eval-all"), (
            "the blind set must stay out of eval-all: scoring it on every retrain "
            "turns it into development signal, which is exactly what it exists to avoid"
        )

    def test_a_dedicated_eval_blind_target_exists(self):
        body = self._target_body("eval-blind")
        assert "text2sql_eval_blind.jsonl" in body

    def test_eval_blind_scores_the_base_model_too(self):
        """Without a base number the blind result has nothing to be compared to."""
        body = self._target_body("eval-blind")
        lines = [ln for ln in body.splitlines() if "eval_baseline" in ln]
        assert len(lines) == 2
        assert any("--adapter" not in ln for ln in lines), "no base-model run"
        assert any("--adapter" in ln for ln in lines), "no adapter run"

    def test_eval_all_still_covers_the_five_dev_sets(self):
        body = self._target_body("eval-all")
        for name in ("text2sql_eval_paraphrase", "text2sql_eval_bookstore",
                     "text2sql_eval_join", "text2sql_eval_join_bookstore"):
            assert name in body, f"eval-all no longer scores {name}"
