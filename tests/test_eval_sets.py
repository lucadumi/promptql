"""Unit tests for the paraphrase (out-of-template) eval set.

The README's honesty caveat is that the main eval is *in-distribution* with the
synthetic training templates. data/eval/text2sql_eval_paraphrase.jsonl re-asks the
same 20 intents (identical gold SQL) with heavily reworded questions to measure
robustness to unseen phrasings. These tests lock in the guarantees that make that
measurement meaningful:

  * it targets the same golds as the in-template eval (only phrasing varies), and
  * its questions are genuinely OOD -- none collide with a training question, so a
    score drop reflects phrasing sensitivity, not memorised strings.

Pure data + SQLite, so the suite stays fast (no torch / model download).
"""
from pathlib import Path

from src.build_dataset import normalize_question
from src.data_utils import load_jsonl
from src.db import build_db, run_sql
from src.metrics import normalize_sql

REPO_ROOT = Path(__file__).resolve().parents[1]
PARAPHRASE = REPO_ROOT / "data" / "eval" / "text2sql_eval_paraphrase.jsonl"
ORIGINAL = REPO_ROOT / "data" / "eval" / "text2sql_eval.jsonl"
TRAIN = REPO_ROOT / "data" / "train" / "text2sql_train.jsonl"
VAL = REPO_ROOT / "data" / "train" / "text2sql_val.jsonl"


class TestParaphraseShape:
    def test_has_twenty_well_formed_rows(self):
        rows = load_jsonl(PARAPHRASE)
        assert len(rows) == 20
        for r in rows:
            assert set(r) >= {"id", "question", "sql"}
            assert r["question"].strip()
            assert r["sql"].strip()

    def test_ids_are_unique(self):
        rows = load_jsonl(PARAPHRASE)
        ids = [r["id"] for r in rows]
        assert len(set(ids)) == len(ids)


class TestControlledVariant:
    def test_golds_match_the_in_template_eval(self):
        # Identical targets -> the only thing that varies is the question wording.
        para = sorted(normalize_sql(r["sql"]) for r in load_jsonl(PARAPHRASE))
        orig = sorted(normalize_sql(r["sql"]) for r in load_jsonl(ORIGINAL))
        assert para == orig


class TestExecutableOnSeedDb:
    def test_every_gold_runs_and_returns_rows(self):
        # Non-empty results keep execution accuracy discriminating on the seed DB.
        conn = build_db()
        for r in load_jsonl(PARAPHRASE):
            assert run_sql(conn, r["sql"]), f"empty/failed gold for id {r['id']}: {r['sql']}"


class TestOutOfTemplate:
    def test_questions_do_not_collide_with_training(self):
        train_q = {
            normalize_question(r["question"]) for r in load_jsonl(TRAIN) + load_jsonl(VAL)
        }
        clashes = [
            r["id"] for r in load_jsonl(PARAPHRASE)
            if normalize_question(r["question"]) in train_q
        ]
        assert clashes == []

    def test_questions_differ_from_the_in_template_eval(self):
        orig_q = {normalize_question(r["question"]) for r in load_jsonl(ORIGINAL)}
        overlap = [
            r["id"] for r in load_jsonl(PARAPHRASE)
            if normalize_question(r["question"]) in orig_q
        ]
        assert overlap == []
