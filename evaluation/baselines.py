"""
baselines.py
------------
Classical baselines the transformers must beat.

A transformer reported in isolation proves nothing: the examiner's
question is always "compared to what?". This module trains three
reference systems per task on exactly the same training split:

  majority   -- always predicts the most frequent class. Establishes the
                floor; on an imbalanced urgency set this can look
                deceptively strong on accuracy, which is precisely why
                macro-F1 is the headline metric everywhere in this
                project.
  tfidf_lr   -- TF-IDF (word 1-2 grams + char 3-5 grams) into logistic
                regression. Strong, fast, interpretable.
  tfidf_svm  -- the same features into a linear SVM.

If DistilBERT cannot beat tfidf_lr by a margin that survives a
significance test, the honest conclusion is that the extra 250MB and the
GPU training run were not justified for this corpus -- and reporting
that is better science than hiding the comparison.

Predictions are cached to evaluation/results/ so evaluate.py can run
McNemar's test between any baseline and the transformer without
retraining anything.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score
from sklearn.pipeline import Pipeline, make_union
from sklearn.svm import LinearSVC

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dataset import RESULTS_DIR, get_splits  # noqa: E402

SEED = 42


def build_features():
    """Word and character n-grams combined.

    Character n-grams matter here because student inquiries contain
    misspellings and inconsistent spacing ("timetabel", "re-reg"), which
    word-level features miss entirely.
    """
    return make_union(
        TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=2,
                        strip_accents="unicode", lowercase=True),
        TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True,
                        min_df=2, lowercase=True),
    )


def build_models():
    return {
        "majority": Pipeline([
            ("features", TfidfVectorizer(min_df=1)),
            ("clf", DummyClassifier(strategy="most_frequent")),
        ]),
        "tfidf_lr": Pipeline([
            ("features", build_features()),
            ("clf", LogisticRegression(max_iter=2000, C=4.0,
                                       class_weight="balanced", random_state=SEED)),
        ]),
        "tfidf_svm": Pipeline([
            ("features", build_features()),
            ("clf", LinearSVC(C=1.0, class_weight="balanced", random_state=SEED)),
        ]),
    }


def run_task(task: str, train: pd.DataFrame, test: pd.DataFrame) -> dict:
    """Fits every baseline for one task and returns their test metrics."""
    print(f"\n--- baselines: {task} ---")
    results = {}
    predictions = {"gold": test[task].tolist(), "text": test["text"].tolist()}

    for name, model in build_models().items():
        model.fit(train["text"], train[task])
        predicted = model.predict(test["text"])
        predictions[name] = list(predicted)

        macro_f1 = f1_score(test[task], predicted, average="macro", zero_division=0)
        weighted_f1 = f1_score(test[task], predicted, average="weighted", zero_division=0)
        accuracy = float(np.mean(np.array(predicted) == test[task].to_numpy()))

        results[name] = {
            "accuracy": accuracy,
            "macro_f1": float(macro_f1),
            "weighted_f1": float(weighted_f1),
            "report": classification_report(
                test[task], predicted, zero_division=0, output_dict=True
            ),
        }
        print(f"  {name:11s} accuracy {accuracy:.4f}  macro-F1 {macro_f1:.4f}")

    pd.DataFrame(predictions).to_csv(
        RESULTS_DIR / f"baseline_predictions_{task}.csv", index=False
    )
    return results


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    train, val, test, _ = get_splits(csv_path)

    if test.empty:
        raise SystemExit("Test split is empty; cannot evaluate baselines.")

    # Baselines get train+val, since they have no validation-set role
    # (no temperature to fit, no early stopping). Giving them the same
    # total supervision as the transformers keeps the comparison fair.
    fitting_set = pd.concat([train, val]).reset_index(drop=True)

    all_results = {task: run_task(task, fitting_set, test)
                   for task in ("category", "urgency")}

    with open(RESULTS_DIR / "baseline_metrics.json", "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)

    summary = []
    for task, models in all_results.items():
        for name, metrics in models.items():
            summary.append({
                "task": task, "model": name,
                "accuracy": round(metrics["accuracy"], 4),
                "macro_f1": round(metrics["macro_f1"], 4),
                "weighted_f1": round(metrics["weighted_f1"], 4),
            })
    table = pd.DataFrame(summary)
    table.to_csv(RESULTS_DIR / "baseline_summary.csv", index=False)

    print("\n" + table.to_string(index=False))
    print(f"\nWrote {RESULTS_DIR / 'baseline_metrics.json'}")


if __name__ == "__main__":
    main()
