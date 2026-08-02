"""Unit tests for src/significance.py -- the interval and paired-test machinery
that every model-vs-model claim in this project now has to pass through. Pure
functions and a seeded RNG, so these are fast and deterministic (no model, no
torch, no numpy).
"""
import json
from pathlib import Path

import pytest

from src.significance import (
    compare_files,
    format_comparison,
    mcnemar_exact,
    paired_bootstrap_diff,
    wilson_interval,
)


class TestWilsonInterval:
    def test_brackets_the_estimate(self):
        low, high = wilson_interval(17, 30)
        assert low < 17 / 30 < high

    def test_stays_inside_zero_and_one_at_the_extremes(self):
        # The textbook normal interval returns bounds outside [0, 1] here, which
        # is the whole reason this project does not use it.
        assert wilson_interval(0, 8) == (0.0, pytest.approx(0.324, abs=0.01))
        low, high = wilson_interval(8, 8)
        assert high == 1.0
        assert low > 0.0

    def test_narrows_as_evidence_grows(self):
        small = wilson_interval(17, 30)
        large = wilson_interval(170, 300)
        assert (large[1] - large[0]) < (small[1] - small[0])

    def test_no_observations_admits_everything(self):
        assert wilson_interval(0, 0) == (0.0, 1.0)


class TestMcNemarExact:
    def test_counts_only_the_disagreements(self):
        # Two questions both get right, two both get wrong: no evidence either
        # way, so they must not appear in the discordant count.
        a = [True, True, False, False, True]
        b = [True, False, False, True, True]
        result = mcnemar_exact(a, b)
        assert result["a_only"] == 1
        assert result["b_only"] == 1
        assert result["discordant"] == 2

    def test_total_agreement_is_not_significant(self):
        result = mcnemar_exact([True, False], [True, False])
        assert result["discordant"] == 0
        assert result["p_value"] == 1.0

    def test_lopsided_split_is_significant(self):
        # Ten disagreements all favouring one model: a fair coin does that with
        # probability 2/2^10.
        a = [True] * 10
        b = [False] * 10
        assert mcnemar_exact(a, b)["p_value"] == pytest.approx(2 / 1024)

    def test_even_split_is_not(self):
        a = [True] * 4 + [False] * 4
        b = [False] * 4 + [True] * 4
        assert mcnemar_exact(a, b)["p_value"] == 1.0

    def test_p_value_never_exceeds_one(self):
        # Doubling the smaller tail can pass 1 without the cap.
        for k in range(9):
            a = [True] * k + [False] * (8 - k)
            b = [False] * k + [True] * (8 - k)
            assert 0.0 <= mcnemar_exact(a, b)["p_value"] <= 1.0

    def test_rejects_unaligned_input(self):
        with pytest.raises(ValueError):
            mcnemar_exact([True], [True, False])


class TestPairedBootstrap:
    def test_is_deterministic_for_a_seed(self):
        a = [True] * 17 + [False] * 13
        b = [True] * 12 + [False] * 18
        first = paired_bootstrap_diff(a, b, n_boot=500, seed=7)
        second = paired_bootstrap_diff(a, b, n_boot=500, seed=7)
        assert first == second

    def test_observed_gap_matches_the_counts(self):
        a = [True] * 6 + [False] * 4
        b = [True] * 3 + [False] * 7
        result = paired_bootstrap_diff(a, b, n_boot=200, seed=0)
        assert result["observed_diff"] == pytest.approx(0.3)

    def test_identical_models_have_no_gap(self):
        a = [True, False, True, False]
        result = paired_bootstrap_diff(a, a, n_boot=200, seed=0)
        assert result["observed_diff"] == 0.0
        assert result["ci_low"] == 0.0 and result["ci_high"] == 0.0

    def test_interval_contains_the_observation(self):
        a = [True] * 17 + [False] * 13
        b = [True] * 12 + [False] * 18
        result = paired_bootstrap_diff(a, b, n_boot=2000, seed=0)
        assert result["ci_low"] <= result["observed_diff"] <= result["ci_high"]

    def test_rejects_unaligned_input(self):
        with pytest.raises(ValueError):
            paired_bootstrap_diff([True], [True, False])


def _write_result(tmp_path: Path, name: str, correct_by_id: dict) -> str:
    """A minimal stand-in for what src/eval_baseline.py writes into results/."""
    path = tmp_path / name
    path.write_text(
        json.dumps(
            {
                "summary": {"n_examples": len(correct_by_id)},
                "records": [
                    {"id": k, "exec_correct": v} for k, v in correct_by_id.items()
                ],
            }
        )
    )
    return str(path)


class TestCompareFiles:
    def test_pairs_on_question_id_not_on_order(self, tmp_path):
        # Same questions, listed in a different order in each file. Pairing by
        # position instead of by id would silently compare different questions.
        a = _write_result(tmp_path, "a.json", {"q1": True, "q2": False, "q3": True})
        b = _write_result(tmp_path, "b.json", {"q3": True, "q1": False, "q2": False})
        result = compare_files(a, b, n_boot=200)
        assert result["n"] == 3
        assert result["mcnemar"]["a_only"] == 1
        assert result["mcnemar"]["b_only"] == 0

    def test_drops_questions_missing_from_either_side(self, tmp_path):
        a = _write_result(tmp_path, "a.json", {"q1": True, "q2": True})
        b = _write_result(tmp_path, "b.json", {"q1": False, "q3": True})
        result = compare_files(a, b, n_boot=200)
        assert result["n"] == 1
        assert result["dropped"] == 2

    def test_refuses_when_nothing_lines_up(self, tmp_path):
        a = _write_result(tmp_path, "a.json", {"q1": True})
        b = _write_result(tmp_path, "b.json", {"q9": True})
        with pytest.raises(ValueError):
            compare_files(a, b, n_boot=200)

    def test_report_names_the_caveat_when_the_interval_spans_zero(self, tmp_path):
        a = _write_result(tmp_path, "a.json", {"q1": True, "q2": True, "q3": False})
        b = _write_result(tmp_path, "b.json", {"q1": True, "q2": False, "q3": False})
        text = format_comparison(compare_files(a, b, n_boot=2000))
        assert "does not establish" in text
        assert "little power" in text


class TestTheHeadlineClaim:
    """The comparison the README leads with, checked against the committed runs.

    The fine-tuned 0.5B scores 17/30 on blind v3 and the 1.5B zero-shot scores
    12/30. That gap is real in the sample and is *not* separable from noise at
    this size, which is why the README quotes an interval beside it. If a future
    change makes this test fail, the README paragraph has to change with it.
    """

    ROOT = Path(__file__).resolve().parents[1]
    LORA = (
        "results/eval_Qwen__Qwen2.5-0.5B-Instruct__lora-qwen2.5-0.5b-joingroup"
        "__text2sql_eval_blind_v3_20260731-145718.json"
    )
    BIGGER = (
        "results/baseline_Qwen__Qwen2.5-1.5B-Instruct"
        "__text2sql_eval_blind_v3_20260731-145942.json"
    )

    def test_the_gap_does_not_clear_zero(self):
        lora, bigger = self.ROOT / self.LORA, self.ROOT / self.BIGGER
        if not (lora.exists() and bigger.exists()):
            pytest.skip("blind v3 result files are not present")

        result = compare_files(str(lora), str(bigger), n_boot=20000, seed=0)

        assert result["n"] == 30
        assert result["a"]["correct"] == 17
        assert result["b"]["correct"] == 12
        assert result["bootstrap"]["observed_diff"] == pytest.approx(1 / 6, abs=1e-9)

        # The claim under test: 57% vs 40% on thirty questions is not a
        # demonstrated win.
        assert result["bootstrap"]["ci_low"] < 0 < result["bootstrap"]["ci_high"]
        assert result["mcnemar"]["p_value"] > 0.05
