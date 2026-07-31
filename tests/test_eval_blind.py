"""Unit tests for the held-back ("blind") eval set and its retired predecessor.

A blind set is a *process* guarantee as much as a data file, and the process has
one failure mode: the moment you read its failures and act on them, it stops
being blind and becomes development signal. This project has now been through
that cycle once, so `data/eval` holds two files rather than one:

  * `text2sql_eval_blind_v1_retired.jsonl` - the first blind set. It was scored
    once, its failures were read, and the construct families they exposed were
    taught. That spends it. It is therefore accounted for as a *development* set:
    it joins the `eval-all` regression loop and is no longer quoted as unbiased.
  * `text2sql_eval_blind_v2.jsonl` - the replacement, and the only file that
    currently carries an unbiased number. It was authored by an independent party
    that was walled off from the training generator, from every existing eval set
    and from all model output, then verified here before anything was scored.

These tests pin the properties that keep that arrangement meaningful:

  * v2 is genuinely fresh - no question or gold overlaps another eval set, and
    nothing collides with what the generator can produce, so de-leaking it costs
    the training set nothing,
  * every gold runs on the seed DB and discriminates,
  * every gold's ordering requirement is *earned* - a gold may only pin row order
    if the question asks for one, or if a LIMIT makes the order load-bearing,
    because `src/db.py::execution_match` grades an ordered gold order-sensitively,
  * v2 stays OUT of the routine `eval-all` regression loop, and
  * v1 is now IN it, which is the mechanical expression of "this one is spent".

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
BLIND = EVAL_DIR / "text2sql_eval_blind_v2.jsonl"
RETIRED = EVAL_DIR / "text2sql_eval_blind_v1_retired.jsonl"
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
        assert len(blind) == 30
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

    def test_every_row_is_tagged_with_a_difficulty_tier(self, blind):
        """The tiers are the independent author's. Results are reported per tier
        because an aggregate over a set this varied hides *which* half of the
        distribution the model fails on."""
        assert {r.get("difficulty") for r in blind} == {"easy", "medium", "hard"}


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

    def test_no_gold_uses_a_function_sqlite_does_not_have(self, blind):
        """`YEAR(hire_date)` is the natural thing to write and does not exist in
        SQLite. A gold containing one would fail to run, which the test above
        catches -- this names the cause rather than just the symptom."""
        for r in blind:
            upper = r["sql"].upper()
            for absent in ("YEAR(", "MONTH(", "DATEPART(", "GETDATE(", "NOW("):
                assert absent not in upper, f"gold {r['id']} uses non-SQLite {absent}"


class TestOrderingIsEarned:
    """`execution_match` compares result sets as a multiset *unless* the gold has
    an ORDER BY, in which case row order must match exactly. A gold that orders
    for cosmetic reasons therefore silently grades a correct answer wrong, which
    would understate the model for a reason that has nothing to do with the model.
    Every ORDER BY here has to be earned by the question or by a LIMIT."""

    ORDER_WORDS = ("first", "order", "sorted", "sort", "ranked", "rank",
                   "highest", "lowest", "top", "descending", "ascending",
                   "cheapest", "busiest", "largest", "smallest", "most", "least")

    def test_every_ordered_gold_is_justified(self, blind):
        unjustified = []
        for r in blind:
            sql = normalize_sql(r["sql"])
            if "order by" not in sql or "limit" in sql:
                continue  # no ordering, or the ordering decides which rows survive
            if not any(w in r["question"].lower() for w in self.ORDER_WORDS):
                unjustified.append(r["id"])
        assert unjustified == [], (
            f"golds {unjustified} pin a row order the question never asks for, "
            "which grades correct-but-unordered predictions as wrong"
        )


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

    def test_no_question_is_reachable_by_the_generator(self, blind):
        """The strong form for questions: not merely absent from the current
        split, but unreachable by the generator at all, under any phrasing or any
        parameter value. So no blind question is a training template in disguise."""
        cq = {normalize_question(q) for _c, q, _s in generate_candidates()}
        assert [r["id"] for r in blind if normalize_question(r["question"]) in cq] == []

    def test_golds_the_generator_can_reach_are_deleted_from_training(self, blind):
        """Golds are the weaker case, and deliberately so. A handful of these
        questions ("what's the average salary in Sales?") have exactly one natural
        SQL answer, which the generator also reaches under other wordings. Rather
        than contort the eval set to dodge that -- which would trade a realistic
        question for a bookkeeping property -- the leakage filter deletes those
        targets from training outright, the same treatment every other eval gold
        gets. The model has to reach them by generalising.

        Note this is where v2 differs from v1: v1 was reachable by nothing, so
        adding it changed zero training examples and its score described an
        already-shipped adapter. v2 costs the training set a few targets, which is
        harmless only because the model is retrained after it is added."""
        cs = {normalize_sql(s) for _c, _q, s in generate_candidates()}
        shared = [r["id"] for r in blind if normalize_sql(r["sql"]) in cs]
        assert shared, (
            "expected a few natural golds to be reachable; if none are, this test "
            "and the README paragraph explaining it are both stale"
        )
        training = {normalize_sql(r["sql"]) for r in load_jsonl(TRAIN) + load_jsonl(VAL)}
        survived = [i for i in shared
                    if normalize_sql(next(r["sql"] for r in blind if r["id"] == i))
                    in training]
        assert survived == [], (
            f"blind golds {survived} are reachable by the generator AND present in "
            "the training data -- the de-leak filter did not fire"
        )


class TestCoverage:
    def test_covers_the_constructs_the_dev_sets_test(self, blind):
        golds = " ".join(r["sql"].upper() for r in blind)
        for kw in ("COUNT(", "AVG(", "MAX(", "MIN(", "DISTINCT", "WHERE",
                   "ORDER BY", "LIMIT", "GROUP BY", "HAVING", "JOIN"):
            assert kw in golds, f"blind set is missing construct: {kw}"

    def test_still_reaches_past_what_the_generator_can_teach(self, blind):
        """v1 reached past the syllabus through single keywords (LIKE, IS NULL,
        BETWEEN, SELECT DISTINCT). Those are now taught, so asserting on them
        would prove nothing. What the generator still cannot produce is
        *composition*: a query whose answer depends on another query. Deriving
        that from the generator itself, rather than from a hardcoded keyword list,
        keeps the test honest as the training set grows."""
        generated = " ".join(normalize_sql(s) for _c, _q, s in generate_candidates())
        assert "select" in generated, "candidate pool is empty"
        assert "(select" not in generated, (
            "the generator now emits subqueries, so this test needs a new "
            "definition of 'past what training teaches'"
        )
        composed = [r["id"] for r in blind if r["sql"].upper().count("SELECT") > 1]
        assert len(composed) >= 5, f"only {len(composed)} composed golds: {composed}"

    def test_includes_a_self_join(self, blind):
        selfjoins = [r["id"] for r in blind
                     if " join employees" in normalize_sql(r["sql"])]
        assert selfjoins, "expected at least one self-join gold"


class TestHeldBackFromTheRegressionLoop:
    """The process guarantee, enforced mechanically rather than by discipline."""

    def _target_body(self, name: str) -> str:
        text = MAKEFILE.read_text()
        match = re.search(rf"^{name}:\n((?:\t.*\n|\n(?=\t))*)", text, re.M)
        assert match, f"no {name} target in the Makefile"
        return match.group(1)

    def test_eval_all_does_not_score_the_current_blind_set(self):
        assert BLIND.name not in self._target_body("eval-all"), (
            "the blind set must stay out of eval-all: scoring it on every retrain "
            "turns it into development signal, which is exactly what it exists to avoid"
        )

    def test_a_dedicated_eval_blind_target_exists(self):
        assert BLIND.name in self._target_body("eval-blind")

    def test_eval_blind_scores_the_base_model_too(self):
        """Without a base number the blind result has nothing to be compared to."""
        body = self._target_body("eval-blind")
        lines = [ln for ln in body.splitlines() if "eval_baseline" in ln]
        assert len(lines) == 2
        assert any("--adapter" not in ln for ln in lines), "no base-model run"
        assert any("--adapter" in ln for ln in lines), "no adapter run"

    def test_eval_all_still_covers_every_development_set(self):
        body = self._target_body("eval-all")
        for name in ("text2sql_eval_paraphrase", "text2sql_eval_bookstore",
                     "text2sql_eval_join", "text2sql_eval_join_bookstore"):
            assert name in body, f"eval-all no longer scores {name}"


class TestRetiredPredecessor:
    """v1 was spent the moment its failures were used to decide what to teach.
    Retiring it means saying so in the file name and moving it into the
    development loop -- not deleting it, and not quietly re-quoting its number as
    though it were still unbiased."""

    def test_the_retired_set_is_still_present(self):
        assert len(load_jsonl(RETIRED)) == 24

    def test_the_retired_set_joined_the_regression_loop(self):
        text = MAKEFILE.read_text()
        match = re.search(r"^eval-all:\n((?:\t.*\n|\n(?=\t))*)", text, re.M)
        assert match and RETIRED.name in match.group(1), (
            "the retired blind set is development signal now and belongs in "
            "eval-all, so a regression on the constructs it exposed cannot pass "
            "unnoticed"
        )

    def test_its_golds_are_still_forbidden_as_training_targets(self):
        """It stays in data/eval, which is what the generator de-leaks against, so
        its golds remain unusable as training targets even though its failures
        were allowed to motivate new construct *families*. That is the line: teach
        the construct, never the graded answer."""
        training = {normalize_sql(r["sql"]) for r in load_jsonl(TRAIN) + load_jsonl(VAL)}
        leaked = [r["id"] for r in load_jsonl(RETIRED)
                  if normalize_sql(r["sql"]) in training]
        assert leaked == [], f"retired-set golds leaked into training: {leaked}"

    def test_the_constructs_it_exposed_are_now_actually_taught(self):
        """The other half of that line: the families really were added. Without
        this, "we taught the constructs" is an unverified claim in the README."""
        training = " ".join(normalize_sql(r["sql"])
                            for r in load_jsonl(TRAIN) + load_jsonl(VAL))
        for construct in ("is null", "is not null", "select distinct",
                          "strftime(", "like '", "between", "!="):
            assert construct in training, f"no training example teaches {construct}"
