"""
stats_tests.py
--------------
Significance testing utilities, implemented directly on scipy so the
project does not take a statsmodels dependency for two functions.

McNemar's exact test compares two classifiers evaluated on the same test
set. Build the 2x2 disagreement table:

                        baseline correct   baseline wrong
    model correct              a                  b
    model wrong                c                  d

Only b and c carry information: the items both systems got right or both
got wrong say nothing about which is better. Under the null hypothesis
that the two systems are equally accurate, each disagreement is a fair
coin flip, so b ~ Binomial(b + c, 0.5) and the exact two-sided p-value
is the binomial test on that. This is preferred over the chi-square
approximation when b + c is small, which it usually is on a few hundred
test items.

Bootstrap confidence intervals are included because a single macro-F1
point estimate on a small test set is fragile, and a CI is what turns
"0.91 vs 0.89" into a defensible claim.
"""

import numpy as np
from scipy import stats
from sklearn.metrics import f1_score


def mcnemar_exact(gold, model_predictions, baseline_predictions) -> dict:
    """Exact McNemar test between two classifiers on the same items."""
    gold = np.asarray(gold)
    model_correct = np.asarray(model_predictions) == gold
    baseline_correct = np.asarray(baseline_predictions) == gold

    a = int(np.sum(model_correct & baseline_correct))
    b = int(np.sum(model_correct & ~baseline_correct))
    c = int(np.sum(~model_correct & baseline_correct))
    d = int(np.sum(~model_correct & ~baseline_correct))

    if b + c == 0:
        # The two systems never disagree; no evidence either way.
        p_value = 1.0
    else:
        p_value = float(stats.binomtest(b, b + c, 0.5, alternative="two-sided").pvalue)

    return {
        "a_both_correct": a,
        "b": b,
        "c": c,
        "d_both_wrong": d,
        "p_value": p_value,
        "significant_at_05": p_value < 0.05,
        "favours": "model" if b > c else ("baseline" if c > b else "tie"),
    }


def bootstrap_macro_f1(gold, predictions, n_resamples: int = 2000,
                       confidence: float = 0.95, seed: int = 42) -> dict:
    """Percentile bootstrap confidence interval for macro-F1.

    Resamples test items with replacement, recomputing macro-F1 each
    time. Reporting the interval alongside the point estimate is what
    stops a small test set from being over-claimed.
    """
    gold = np.asarray(gold)
    predictions = np.asarray(predictions)
    rng = np.random.default_rng(seed)
    n = len(gold)

    point = f1_score(gold, predictions, average="macro", zero_division=0)
    scores = []
    for _ in range(n_resamples):
        indices = rng.integers(0, n, n)
        resampled_gold = gold[indices]
        if len(np.unique(resampled_gold)) < 2:
            continue
        scores.append(
            f1_score(resampled_gold, predictions[indices],
                     average="macro", zero_division=0)
        )

    if not scores:
        return {"macro_f1": float(point), "ci_low": None, "ci_high": None}

    alpha = (1 - confidence) / 2
    return {
        "macro_f1": float(point),
        "ci_low": float(np.quantile(scores, alpha)),
        "ci_high": float(np.quantile(scores, 1 - alpha)),
        "confidence": confidence,
        "n_resamples": len(scores),
    }
