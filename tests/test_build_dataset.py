"""Tests for src/build_dataset.py.

The single most important correctness property of this project is: **no training
example may leak into the held-out eval set** (same normalized question or SQL).
The 40% -> 100% result is only honest if that holds. These tests prove it on the
actually-generated data, plus reproducibility and stratification.

Everything here is stdlib-only (no torch), so it runs in milliseconds.
"""
from src.build_dataset import (
    DEFAULT_EVAL,
    build,
    generate_candidates,
    normalize_question,
    verify_no_leakage,
    write_jsonl,
)
from src.data_utils import load_jsonl
from src.metrics import normalize_sql

SEED = 13
VAL_FRAC = 0.15


def _build():
    return build(DEFAULT_EVAL, val_frac=VAL_FRAC, seed=SEED)


def test_build_produces_nonempty_splits():
    rep = _build()
    assert len(rep["train"]) > 0
    assert len(rep["val"]) > 0


def test_no_leakage_between_generated_data_and_eval():
    """THE invariant: nothing we generate collides with the eval set."""
    rep = _build()
    eval_questions, eval_sqls = rep["eval_questions"], rep["eval_sqls"]
    for cat, q, sql in rep["train"] + rep["val"]:
        assert normalize_question(q) not in eval_questions, f"leaked question ({cat}): {q}"
        assert normalize_sql(sql) not in eval_sqls, f"leaked sql ({cat}): {sql}"


def test_verify_no_leakage_helper_does_not_raise():
    rep = _build()
    # The generator ships its own belt-and-braces assertion; make sure it agrees.
    verify_no_leakage(rep["train"] + rep["val"], rep["eval_questions"], rep["eval_sqls"])


def test_questions_are_unique_across_splits():
    rep = _build()
    questions = [normalize_question(q) for _cat, q, _sql in rep["train"] + rep["val"]]
    assert len(questions) == len(set(questions)), "duplicate normalized questions found"


def test_build_is_reproducible_for_a_fixed_seed():
    a, b = _build(), _build()
    assert sorted(a["train"]) == sorted(b["train"])
    assert sorted(a["val"]) == sorted(b["val"])


def test_every_category_keeps_at_least_one_training_example():
    """Stratified split must never move a whole pattern into val (train-only gap)."""
    rep = _build()
    train_cats = {cat for cat, _q, _sql in rep["train"]}
    assert set(rep["by_cat"]).issubset(train_cats)


def test_dropped_counts_are_consistent():
    rep = _build()
    # kept + dropped(leak) + dropped(dup) accounts for every candidate generated.
    assert rep["n_kept"] + rep["dropped_leak"] + rep["dropped_dup"] == rep["n_candidates"]
    assert rep["n_kept"] == len(rep["train"]) + len(rep["val"])


def test_generate_candidates_returns_cat_question_sql_triples():
    cands = generate_candidates()
    assert len(cands) > 0
    for item in cands[:5]:
        assert len(item) == 3
        cat, q, sql = item
        assert isinstance(cat, str) and isinstance(q, str) and isinstance(sql, str)


def test_written_records_use_eval_compatible_schema(tmp_path):
    rep = _build()
    out = tmp_path / "train.jsonl"
    write_jsonl(rep["train"], out)
    rows = load_jsonl(out)
    assert len(rows) == len(rep["train"])
    assert set(rows[0]) == {"id", "question", "sql"}
