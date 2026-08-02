"""Significance testing for eval-set comparisons.

Every accuracy in this project is a count out of thirty-something questions, and
a count that small carries an interval wide enough to swallow most of the gaps
worth arguing about. Quoting `57% vs 40%` without one invites the reader to
believe a difference the data may not support.

This module supplies the three things missing from that sentence:

* `wilson_interval` -- how precise a single accuracy actually is. The textbook
  `p +/- 1.96*sqrt(p(1-p)/n)` is badly behaved at these sample sizes and near 0
  or 1, where it happily returns bounds outside [0, 1]. Wilson's interval does
  not.
* `mcnemar_exact` -- whether two models scored on the *same* questions really
  differ. Because the questions are shared, the only informative examples are
  the ones where the two models disagree; the ones they both get right or both
  get wrong carry no evidence either way. That reduces to a sign test on the
  disagreements, computed exactly from the binomial rather than through the
  chi-squared approximation, which is not trustworthy when there is a handful of
  them.
* `paired_bootstrap_diff` -- an interval on the gap itself, resampling questions
  with the pairing intact so the two models are always scored on the same draw.

Deliberately stdlib-only. `make test` is meant to run without downloading a
model, and a significance test that drags in a numerical stack to compute a
binomial tail would be a poor trade.

The unit here is one question, and questions are assumed independent. That is
reasonable for these eval sets, where each was written as a separate item, and
it is *not* the same assumption sibyl makes about overlapping time series -- see
`bootstrap_sharpe_diff` there, which resamples blocks for exactly that reason.
"""
from __future__ import annotations

import json
import random
from math import comb, sqrt
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

# 95 percent, two-sided, from the normal distribution.
Z_95 = 1.959963984540054


def wilson_interval(successes: int, n: int, z: float = Z_95) -> Tuple[float, float]:
    """A confidence interval for one accuracy, well behaved at small n.

    Returns (low, high) as proportions. With no observations at all the interval
    is the whole range, which is the honest answer rather than an error.
    """
    if n <= 0:
        return (0.0, 1.0)

    p = successes / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    margin = (z / denom) * sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (max(0.0, centre - margin), min(1.0, centre + margin))


def _binomial_two_sided(k: int, n: int) -> float:
    """Two-sided p-value for k successes in n fair coin flips.

    Doubling the smaller tail. For the symmetric null used here (a disagreement
    is equally likely to favour either model) that is exact rather than an
    approximation, and it is capped at 1 because doubling a tail past a half
    would otherwise exceed it.
    """
    if n == 0:
        return 1.0
    tail = sum(comb(n, i) for i in range(0, min(k, n - k) + 1))
    return min(1.0, 2.0 * tail / (2 ** n))


def mcnemar_exact(a_correct: Sequence[bool], b_correct: Sequence[bool]) -> Dict[str, float]:
    """Exact paired test for two models scored on the same questions.

    `a_correct` and `b_correct` must be aligned: element i is the same question
    for both. Only the disagreements carry information, so the test asks whether
    they split more unevenly than a fair coin would explain.

    Returns the two counts, the p-value, and the number of questions. A p-value
    near 1 with few disagreements means the comparison is underpowered, not that
    the models are equal, and the caller should say so.
    """
    if len(a_correct) != len(b_correct):
        raise ValueError("paired test needs equal-length, aligned sequences")

    a_only = sum(1 for a, b in zip(a_correct, b_correct) if a and not b)
    b_only = sum(1 for a, b in zip(a_correct, b_correct) if b and not a)

    return {
        "n": len(a_correct),
        "a_only": a_only,
        "b_only": b_only,
        "discordant": a_only + b_only,
        "p_value": _binomial_two_sided(a_only, a_only + b_only),
    }


def paired_bootstrap_diff(
    a_correct: Sequence[bool],
    b_correct: Sequence[bool],
    n_boot: int = 20000,
    seed: int = 0,
) -> Dict[str, float]:
    """Interval on the accuracy gap, resampling questions in pairs.

    Questions are drawn with replacement and both models are scored on the same
    draw, so the resampling never breaks the pairing that makes the comparison
    sharp. Returns the observed gap (positive means `a` is ahead), a 95 percent
    interval, and how often `a` came out ahead across resamples.

    An interval that contains zero is the useful case: it says the observed gap
    is inside what resampling the same questions could have produced anyway.
    """
    if len(a_correct) != len(b_correct):
        raise ValueError("paired bootstrap needs equal-length, aligned sequences")

    n = len(a_correct)
    if n == 0:
        return {}

    a = [1 if x else 0 for x in a_correct]
    b = [1 if x else 0 for x in b_correct]
    observed = (sum(a) - sum(b)) / n

    rng = random.Random(seed)
    diffs: List[float] = []
    for _ in range(n_boot):
        total = 0
        for _ in range(n):
            i = rng.randrange(n)
            total += a[i] - b[i]
        diffs.append(total / n)

    diffs.sort()
    lo = diffs[int(0.025 * len(diffs))]
    hi = diffs[min(int(0.975 * len(diffs)), len(diffs) - 1)]

    return {
        "observed_diff": observed,
        "ci_low": lo,
        "ci_high": hi,
        "prob_a_ahead": sum(1 for d in diffs if d > 0) / len(diffs),
    }


def load_records(path: str) -> Dict[str, bool]:
    """Read one eval result file into {question id: was it execution-correct}.

    These are the JSON files `src/eval_baseline.py` writes into `results/`.
    """
    payload = json.loads(Path(path).read_text())
    return {r["id"]: bool(r["exec_correct"]) for r in payload["records"]}


def compare_files(
    path_a: str, path_b: str, n_boot: int = 20000, seed: int = 0
) -> Dict[str, object]:
    """Compare two result files question by question.

    Aligns on question id and drops anything missing from either side, because a
    paired test on unaligned questions is not a paired test.
    """
    a_all, b_all = load_records(path_a), load_records(path_b)
    ids = sorted(set(a_all) & set(b_all))
    if not ids:
        raise ValueError("the two result files share no question ids")

    a = [a_all[i] for i in ids]
    b = [b_all[i] for i in ids]
    n = len(ids)

    return {
        "n": n,
        "dropped": len(set(a_all) ^ set(b_all)),
        "a": {
            "file": path_a,
            "correct": sum(a),
            "accuracy": sum(a) / n,
            "ci": wilson_interval(sum(a), n),
        },
        "b": {
            "file": path_b,
            "correct": sum(b),
            "accuracy": sum(b) / n,
            "ci": wilson_interval(sum(b), n),
        },
        "mcnemar": mcnemar_exact(a, b),
        "bootstrap": paired_bootstrap_diff(a, b, n_boot=n_boot, seed=seed),
    }


def format_comparison(result: Dict[str, object]) -> str:
    """Render a comparison as the paragraph a README should quote."""
    a, b = result["a"], result["b"]
    mc, bs = result["mcnemar"], result["bootstrap"]
    n = result["n"]

    lines = [
        f"n = {n} shared questions"
        + (f" ({result['dropped']} dropped as unmatched)" if result["dropped"] else ""),
        "",
        f"  A  {Path(a['file']).name}",
        f"     {a['correct']}/{n} = {a['accuracy']:.1%}"
        f"   95% CI [{a['ci'][0]:.1%}, {a['ci'][1]:.1%}]",
        f"  B  {Path(b['file']).name}",
        f"     {b['correct']}/{n} = {b['accuracy']:.1%}"
        f"   95% CI [{b['ci'][0]:.1%}, {b['ci'][1]:.1%}]",
        "",
        f"  gap (A - B)          {bs['observed_diff']:+.1%}",
        f"  paired bootstrap CI  [{bs['ci_low']:+.1%}, {bs['ci_high']:+.1%}]",
        f"  disagreements        {mc['a_only']} favour A, {mc['b_only']} favour B",
        f"  exact McNemar p      {mc['p_value']:.4f}",
    ]

    # The interpretation, so a number cannot be quoted without the caveat that
    # belongs to it.
    crosses_zero = bs["ci_low"] <= 0 <= bs["ci_high"]
    if crosses_zero:
        lines += [
            "",
            "  The interval contains zero: this data does not establish that A",
            "  beats B. Report the gap with the interval, or not at all.",
        ]
    else:
        lines += ["", "  The interval excludes zero: the gap survives resampling."]

    if mc["discordant"] < 10:
        lines += [
            f"  Only {mc['discordant']} questions separate the two models, so this",
            "  comparison has little power either way.",
        ]

    return "\n".join(lines)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Compare two eval result files with a paired significance test."
    )
    parser.add_argument("file_a", help="results/*.json for the model under test")
    parser.add_argument("file_b", help="results/*.json for the model to compare against")
    parser.add_argument("--n-boot", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--json", action="store_true", help="emit the raw numbers")
    args = parser.parse_args()

    result = compare_files(args.file_a, args.file_b, n_boot=args.n_boot, seed=args.seed)
    print(json.dumps(result, indent=2) if args.json else format_comparison(result))


if __name__ == "__main__":
    main()
