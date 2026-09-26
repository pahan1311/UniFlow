"""
evaluate.py
-----------
Held-out evaluation of the fine-tuned transformers, with confusion
matrices and a significance test against the classical baselines.

Produces, per task:
  * accuracy, macro-F1, weighted-F1 and a full per-class
    precision/recall/F1/support table
  * a confusion matrix, saved both as CSV and as a labelled PNG
  * McNemar's test against each baseline

Why McNemar and not a t-test: both systems are evaluated on the *same*
test items, so the observations are paired, and the quantity of interest
is the disagreement counts -- how often the transformer is right where
the baseline is wrong (b) versus the reverse (c). McNemar's exact test
on (b, c) is the standard choice for comparing two classifiers on one
test set (Dietterich, 1998). A difference in macro-F1 with p > 0.05
should be reported as "no significant difference", not as a win.

Logits are cached to evaluation/results/ so calibrate.py, selective.py
and ood.py never have to re-run the models.
"""

import gc
import json
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))

from dataset import RESULTS_DIR, get_splits  # noqa: E402
from logits_cache import cached_logits as compute_logits  # noqa: E402
from logits_cache import labels_for  # noqa: E402
from stats_tests import bootstrap_macro_f1, mcnemar_exact  # noqa: E402


def plot_confusion(matrix, labels, title, output_path, normalise=True):
    """Saves a labelled confusion-matrix figure.

    Row-normalised by default: with imbalanced classes the raw-count
    version is dominated by the largest class and hides exactly the
    per-class failures the matrix exists to reveal.
    """
    display = matrix.astype(float)
    if normalise:
        row_sums = display.sum(axis=1, keepdims=True)
        display = np.divide(display, row_sums, out=np.zeros_like(display),
                            where=row_sums != 0)

    size = max(5.0, 0.75 * len(labels) + 2.5)
    fig, ax = plt.subplots(figsize=(size, size * 0.85))
    image = ax.imshow(display, cmap="Greens", vmin=0, vmax=1 if normalise else None)

    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=40, ha="right", fontsize=8)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)

    threshold = display.max() / 2 if display.max() else 0.5
    for i in range(len(labels)):
        for j in range(len(labels)):
            cell = f"{display[i, j]:.2f}" if normalise else f"{int(matrix[i, j])}"
            ax.text(j, i, cell, ha="center", va="center", fontsize=7,
                    color="white" if display[i, j] > threshold else "#173d2c")

    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def evaluate_task(task, test_df, labels, baseline_predictions=None):
    """Full metric set for one task, plus significance tests."""
    print(f"\n--- transformer: {task} ---")
    logits = compute_logits(test_df["text"], task, "test")
    predicted = [labels[i] for i in logits.argmax(axis=1)]
    gold = test_df[task].tolist()

    unseen = sorted(set(gold) - set(labels))
    if unseen:
        print(f"  ! test set contains labels the model cannot predict: {unseen}")

    accuracy = accuracy_score(gold, predicted)
    macro_f1 = f1_score(gold, predicted, average="macro", zero_division=0)
    weighted_f1 = f1_score(gold, predicted, average="weighted", zero_division=0)
    interval = bootstrap_macro_f1(gold, predicted)
    print(f"  accuracy {accuracy:.4f}  macro-F1 {macro_f1:.4f}  "
          f"weighted-F1 {weighted_f1:.4f}")
    if interval["ci_low"] is not None:
        print(f"  macro-F1 95% CI [{interval['ci_low']:.4f}, "
              f"{interval['ci_high']:.4f}]  (bootstrap, "
              f"{interval['n_resamples']} resamples)")

    report = classification_report(gold, predicted, zero_division=0, output_dict=True)
    per_class = pd.DataFrame(report).transpose().round(4)
    per_class.to_csv(RESULTS_DIR / f"per_class_{task}.csv")
    print("\n" + classification_report(gold, predicted, zero_division=0))

    matrix_labels = sorted(set(gold) | set(predicted))
    matrix = confusion_matrix(gold, predicted, labels=matrix_labels)
    pd.DataFrame(matrix, index=matrix_labels, columns=matrix_labels).to_csv(
        RESULTS_DIR / f"confusion_{task}.csv"
    )
    plot_confusion(
        matrix, matrix_labels,
        f"{task.title()} confusion matrix (row-normalised)",
        RESULTS_DIR / f"confusion_{task}.png",
    )

    significance = {}
    if baseline_predictions is not None:
        for name in ("majority", "tfidf_lr", "tfidf_svm"):
            if name not in baseline_predictions.columns:
                continue
            baseline = baseline_predictions[name].tolist()
            statistic = mcnemar_exact(gold, predicted, baseline)
            significance[name] = statistic
            verdict = (
                "significant" if statistic["p_value"] < 0.05 else "NOT significant"
            )
            print(f"  vs {name:11s}: transformer-only-correct={statistic['b']:4d} "
                  f"baseline-only-correct={statistic['c']:4d} "
                  f"p={statistic['p_value']:.4g} ({verdict})")

    return {
        "accuracy": float(accuracy),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "macro_f1_ci": interval,
        "per_class": report,
        "mcnemar": significance,
        "n_test": len(gold),
    }


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    train, val, test, leakage_risk = get_splits(csv_path)
    if test.empty:
        raise SystemExit("Test split is empty; nothing to evaluate.")

    from policy_files import MODELS_DIR  # noqa: PLC0415
    backend = ("multitask"
               if (MODELS_DIR / "multitask" / "multitask_config.json").exists()
               else "dual")
    print(f"\nBackend: {backend}")

    results = {"backend": backend, "leakage_risk": leakage_risk, "tasks": {}}

    # One task at a time: each task's model is loaded, used for both the
    # validation and test splits, then released before the next task's
    # model is loaded. Keeping both resident at once is what exhausts
    # memory on a modest laptop.
    for task in ("category", "urgency"):
        if not val.empty:
            # Cached here so calibrate.py, which fits the temperature on
            # validation, never has to load a model at all.
            compute_logits(val["text"], task, "val")

        baseline_path = RESULTS_DIR / f"baseline_predictions_{task}.csv"
        baseline_predictions = (
            pd.read_csv(baseline_path) if baseline_path.exists() else None
        )
        if baseline_predictions is None:
            print(f"\n  (no baseline predictions for {task}; run baselines.py first "
                  f"to enable significance testing)")

        results["tasks"][task] = evaluate_task(
            task, test, labels_for(task), baseline_predictions
        )
        gc.collect()

    with open(RESULTS_DIR / "transformer_metrics.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\nWrote {RESULTS_DIR / 'transformer_metrics.json'}")
    if leakage_risk:
        print("NOTE: split was generated, not supplied. See the leakage warning above.")


if __name__ == "__main__":
    main()
