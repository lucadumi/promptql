"""Tests for src/build_dataset.py.

The single most important correctness property of this project is: **no training
example may leak into any held-out eval set** (same normalized question or SQL).
The 40% -> 100% result is only honest if that holds. These tests prove it on the
actually-generated data, plus reproducibility, stratification, and the phrasing
diversity that the paraphrase-robustness fine-tune depends on.

Everything here is stdlib-only (no torch), so it runs in milliseconds.
"""
from src.build_dataset import (
    IDENTICAL_OVERLAP,
    build,
    closest_eval_question,
    default_eval_files,
    generate_candidates,
    jaccard,
    normalize_question,
    question_words,
    verify_no_leakage,
    write_jsonl,
)
from src.data_utils import load_jsonl
from src.metrics import normalize_sql

SEED = 13
VAL_FRAC = 0.15


def _build():
    return build(default_eval_files(), val_frac=VAL_FRAC, seed=SEED)


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
    verify_no_leakage(rep["train"] + rep["val"], rep["eval_questions"], rep["eval_sqls"],
                      rep["eval_word_sets"])


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
    # every candidate is either kept, leaked, duplicated, or cut by the cap.
    assert (rep["n_kept"] + rep["dropped_leak"] + rep["dropped_dup"]
            + rep["dropped_cap"]) == rep["n_candidates"]
    assert rep["n_kept"] == len(rep["train"]) + len(rep["val"])
    assert rep["n_before_balance"] == rep["n_kept"] + rep["dropped_cap"]


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


class TestLeakageGuardCoversEveryEvalSet:
    """The blocklist must be built from all of data/eval, not just the base set."""

    def test_default_eval_files_include_all_three_sets(self):
        names = {p.name for p in default_eval_files()}
        assert {"text2sql_eval.jsonl",
                "text2sql_eval_paraphrase.jsonl",
                "text2sql_eval_bookstore.jsonl"} <= names

    def test_no_training_question_appears_in_any_eval_set(self):
        rep = _build()
        train_questions = {normalize_question(q) for _c, q, _s in rep["train"] + rep["val"]}
        for path in default_eval_files():
            for row in load_jsonl(path):
                assert normalize_question(row["question"]) not in train_questions, (
                    f"{path.name} question leaked into training: {row['question']}"
                )

    def test_paraphrase_eval_questions_are_not_reachable_from_the_generator(self):
        """Augmentation must not accidentally reinvent the OOD eval wordings."""
        generated = {normalize_question(q) for _c, q, _s in generate_candidates()}
        paraphrase = [p for p in default_eval_files() if "paraphrase" in p.name][0]
        for row in load_jsonl(paraphrase):
            assert normalize_question(row["question"]) not in generated


class TestPhrasingDiversity:
    """The whole point of the augmentation: many questions per SQL target."""

    def test_most_sql_targets_have_several_phrasings(self):
        """Balancing trims some depth, but most targets keep multiple wordings."""
        rep = _build()
        by_sql = {}
        for _cat, q, sql in rep["train"] + rep["val"]:
            by_sql.setdefault(normalize_sql(sql), set()).add(normalize_question(q))
        multi = [s for s, qs in by_sql.items() if len(qs) > 1]
        assert len(multi) / len(by_sql) > 0.7, "most SQL targets should be asked several ways"

    def test_average_phrasings_per_target_is_reported_and_above_two(self):
        rep = _build()
        assert rep["phrasings_per_sql"] > 2.0
        assert rep["n_sql_targets"] > 0

    def test_phrasings_of_one_target_are_lexically_different(self):
        """Diversity must be real wording change, not just punctuation."""
        rep = _build()
        by_sql = {}
        for _cat, q, sql in rep["train"] + rep["val"]:
            by_sql.setdefault(normalize_sql(sql), []).append(q)
        target = max(by_sql.values(), key=len)
        first_words = {q.split()[0].lower() for q in target}
        assert len(first_words) > 1, f"all phrasings start the same way: {target}"

    def test_training_set_is_substantially_larger_than_before_augmentation(self):
        rep = _build()
        assert rep["n_kept"] > 250


class TestCategoryBalance:
    """No single pattern may dominate: that biases rarer patterns' answers."""

    def test_no_category_exceeds_the_cap(self):
        rep = _build()
        for cat, n in rep["by_cat"].items():
            assert n <= rep["cap"], f"{cat} has {n} examples, above the cap {rep['cap']}"

    def test_balancing_reduces_the_spread_between_patterns(self):
        rep = _build()
        counts = list(rep["by_cat"].values())
        assert max(counts) / min(counts) < 8, "pattern mix is still heavily skewed"

    def test_balance_is_applied_and_reported(self):
        rep = _build()
        assert rep["dropped_cap"] > 0
        assert rep["n_before_balance"] > rep["n_kept"]

    def test_capped_categories_keep_more_than_one_phrasing(self):
        """Round-robin sampling must not collapse a pattern to one wording."""
        rep = _build()
        capped = [c for c, n in rep["by_cat"].items() if n == rep["cap"]]
        assert capped, "expected at least one capped category"
        for cat in capped:
            questions = [q for c, q, _s in rep["train"] + rep["val"] if c == cat]
            starts = {q.split()[0].lower() for q in questions}
            assert len(starts) > 1, f"{cat} collapsed to a single phrasing"

    def test_capping_preserves_every_sql_target_of_a_capped_category(self):
        """Parameter coverage comes first, so no literal is dropped entirely."""
        rep = _build()
        kept_sql = {normalize_sql(s) for _c, _q, s in rep["train"] + rep["val"]}
        all_sql = {normalize_sql(s) for _c, _q, s in generate_candidates()}
        leaked = rep["eval_sqls"]
        missing = {s for s in all_sql - kept_sql if s not in leaked}
        assert not missing, f"cap removed SQL targets entirely: {sorted(missing)[:3]}"

    def test_cap_is_configurable(self):
        small = build(default_eval_files(), val_frac=VAL_FRAC, seed=SEED, cap=5)
        assert max(small["by_cat"].values()) <= 5
        assert small["n_kept"] < _build()["n_kept"]


class TestOverlapReporting:
    """The 'not even close' claim is measured, not asserted."""

    def test_jaccard_bounds(self):
        a, b = question_words("how many employees are there"), question_words("how many are there")
        assert jaccard(a, a) == 1.0
        assert 0.0 < jaccard(a, b) < 1.0
        assert jaccard(frozenset(), a) == 0.0

    def test_no_kept_question_is_a_word_for_word_match_of_an_eval_question(self):
        rep = _build()
        for _cat, q, _sql in rep["train"] + rep["val"]:
            overlap, near = closest_eval_question(q, rep["eval_word_sets"])
            assert overlap < IDENTICAL_OVERLAP, f"near-duplicate of eval: {q!r} vs {near!r}"

    def test_report_exposes_the_closest_pair(self):
        rep = _build()
        assert 0.0 < rep["worst_overlap"] < IDENTICAL_OVERLAP
        train_q, eval_q = rep["worst_pair"]
        assert train_q and eval_q
        assert normalize_question(train_q) != normalize_question(eval_q)
