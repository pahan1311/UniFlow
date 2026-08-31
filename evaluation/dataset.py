"""
dataset.py
----------
Loading and splitting the labelled inquiry corpus.

Expected input: one CSV at data/inquiries.csv with a text column and two
label columns. Column names are matched case-insensitively against a set
of aliases, so an export straight from the Colab notebook usually works
without editing:

    text      <- text | student_query | query | inquiry | message
    category  <- category | department | label | category_label
    urgency   <- urgency | priority | urgency_label

If the CSV already carries a `split` column with values train/val/test,
that split is honoured exactly. THIS IS THE IMPORTANT CASE: if the
models were fine-tuned on rows that later land in the test set, every
reported score is inflated by leakage and the evaluation chapter is
worthless. Export the split you actually trained with wherever possible.

When no `split` column exists the module falls back to a deterministic
stratified split (seed 42) and prints a prominent warning, because it
cannot verify that the resulting test set was unseen during training.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "evaluation" / "results"
DEFAULT_CSV = DATA_DIR / "inquiries.csv"

TEXT_ALIASES = ["text", "student_query", "query", "inquiry", "message", "body"]
CATEGORY_ALIASES = ["category", "department", "label", "category_label", "dept"]
URGENCY_ALIASES = ["urgency", "priority", "urgency_label", "urgency_level"]

SEED = 42
VAL_FRACTION = 0.15
TEST_FRACTION = 0.15


def _resolve_column(df: pd.DataFrame, aliases, purpose: str) -> str:
    lowered = {c.lower().strip(): c for c in df.columns}
    for alias in aliases:
        if alias in lowered:
            return lowered[alias]
    raise SystemExit(
        f"\nCould not find a {purpose} column in the CSV.\n"
        f"Columns present: {list(df.columns)}\n"
        f"Expected one of: {aliases}\n"
        f"Rename the column in your CSV, or add the name to "
        f"{purpose.upper()}_ALIASES in evaluation/dataset.py.\n"
    )


def load_corpus(csv_path: Path | None = None) -> pd.DataFrame:
    """Reads the corpus and normalises it to text/category/urgency[/split]."""
    csv_path = Path(csv_path or DEFAULT_CSV)
    if not csv_path.exists():
        raise SystemExit(
            f"\nDataset not found at {csv_path}.\n\n"
            f"Export the labelled dataset you trained on from Colab as CSV and save "
            f"it there. It needs a text column plus category and urgency labels, and "
            f"ideally a 'split' column recording the train/val/test assignment used "
            f"during training.\n"
        )

    df = pd.read_csv(csv_path)
    text_col = _resolve_column(df, TEXT_ALIASES, "text")
    category_col = _resolve_column(df, CATEGORY_ALIASES, "category")
    urgency_col = _resolve_column(df, URGENCY_ALIASES, "urgency")

    normalised = pd.DataFrame({
        "text": df[text_col].astype(str).str.strip(),
        "category": df[category_col].astype(str).str.strip(),
        "urgency": df[urgency_col].astype(str).str.strip(),
    })
    if "split" in {c.lower() for c in df.columns}:
        split_col = {c.lower(): c for c in df.columns}["split"]
        normalised["split"] = df[split_col].astype(str).str.lower().str.strip()

    before = len(normalised)
    normalised = normalised[normalised["text"].str.len() > 0]
    normalised = normalised.drop_duplicates(subset=["text"])
    if len(normalised) < before:
        print(f"  removed {before - len(normalised)} empty or duplicate rows")

    return normalised.reset_index(drop=True)


def stratified_split(df: pd.DataFrame, seed: int = SEED) -> pd.DataFrame:
    """Deterministic stratified split on the joint (category, urgency) label.

    Stratifying on the pair rather than on category alone keeps the
    urgency distribution comparable across splits, which matters because
    urgency is the more imbalanced of the two tasks.
    """
    rng = np.random.default_rng(seed)
    df = df.copy()
    df["split"] = "train"

    joint = df["category"] + "||" + df["urgency"]
    for group in joint.unique():
        # copy=True because a pandas Index exposes a read-only array, and
        # rng.shuffle needs to write in place.
        indices = df.index[joint == group].to_numpy(copy=True)
        rng.shuffle(indices)
        n = len(indices)
        n_test = max(1, int(round(n * TEST_FRACTION))) if n >= 4 else 0
        n_val = max(1, int(round(n * VAL_FRACTION))) if n >= 4 else 0
        df.loc[indices[:n_test], "split"] = "test"
        df.loc[indices[n_test:n_test + n_val], "split"] = "val"

    return df


def get_splits(csv_path: Path | None = None, quiet: bool = False):
    """Returns (train_df, val_df, test_df) plus a leakage warning flag."""
    df = load_corpus(csv_path)

    provided = "split" in df.columns and df["split"].isin(
        {"train", "val", "test", "validation", "dev"}
    ).all()

    if provided:
        df["split"] = df["split"].replace({"validation": "val", "dev": "val"})
        leakage_risk = False
    else:
        df = stratified_split(df)
        leakage_risk = True
        if not quiet:
            print(
                "\n" + "!" * 74 + "\n"
                "  WARNING: the CSV has no 'split' column, so a fresh stratified\n"
                "  split was generated. If your models were fine-tuned on rows that\n"
                "  now fall in this test set, the reported scores are optimistically\n"
                "  biased by data leakage and MUST NOT be presented as held-out\n"
                "  results. Export the original split from your training notebook,\n"
                "  or retrain on the train split produced here.\n"
                + "!" * 74 + "\n"
            )

    train = df[df["split"] == "train"].reset_index(drop=True)
    val = df[df["split"] == "val"].reset_index(drop=True)
    test = df[df["split"] == "test"].reset_index(drop=True)

    if not quiet:
        print(f"  corpus: {len(df)} rows -> train {len(train)}, val {len(val)}, "
              f"test {len(test)}")
        if val.empty or test.empty:
            print("  ! validation or test split is empty; check the dataset size")

    return train, val, test, leakage_risk


def describe(df: pd.DataFrame) -> pd.DataFrame:
    """Label distribution table for the dataset section of the report."""
    rows = []
    for task in ("category", "urgency"):
        counts = df[task].value_counts()
        for label, count in counts.items():
            rows.append({
                "task": task, "label": label, "count": int(count),
                "share": count / len(df),
            })
    return pd.DataFrame(rows)


def main():
    """Writes the splits and a dataset description to evaluation/results/."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = Path(sys.argv[1]) if len(sys.argv) > 1 else None

    train, val, test, leakage_risk = get_splits(csv_path)
    full = pd.concat([
        train.assign(split="train"), val.assign(split="val"), test.assign(split="test")
    ])
    full.to_csv(RESULTS_DIR / "splits.csv", index=False)

    description = describe(full)
    description.to_csv(RESULTS_DIR / "label_distribution.csv", index=False)

    print("\nLabel distribution")
    print(description.to_string(index=False))
    print(f"\nWrote {RESULTS_DIR / 'splits.csv'}")
    if leakage_risk:
        print("Remember to address the leakage warning above in your methodology.")


if __name__ == "__main__":
    main()
