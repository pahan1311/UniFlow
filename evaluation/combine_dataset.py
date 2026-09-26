"""
combine_dataset.py
-------------------
Combines the three separately-exported train/validation/test CSVs into
one data/inquiries.csv with a `split` column, which is the input shape
dataset.py (and therefore run_all.py) expects.

This exists because the split honoured here must be the exact one the
models were fine-tuned on -- letting the evaluation pipeline generate
its own split instead would risk scoring the transformer on rows it
was already trained on. See the leakage warning in dataset.py.

Usage:
    python evaluation/combine_dataset.py \
        --train data/university_issue_train_text_only.csv \
        --val data/university_issue_validation_text_only.csv \
        --test data/university_issue_test_text_only.csv
"""

import argparse
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_OUT = BASE_DIR / "data" / "inquiries.csv"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", required=True)
    parser.add_argument("--val", required=True)
    parser.add_argument("--test", required=True)
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    args = parser.parse_args()

    frames = []
    for split_name, path in (("train", args.train), ("val", args.val), ("test", args.test)):
        df = pd.read_csv(path)
        df["split"] = split_name
        frames.append(df)
        print(f"{split_name}: {len(df)} rows from {path}")

    combined = pd.concat(frames, ignore_index=True)

    duplicate_ids = combined["Query_ID"].duplicated().sum() if "Query_ID" in combined.columns else 0
    if duplicate_ids:
        print(f"WARNING: {duplicate_ids} duplicate Query_ID values across the combined files.")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(out_path, index=False)

    print(f"\nWrote {len(combined)} rows to {out_path}")
    print(combined["split"].value_counts().to_string())


if __name__ == "__main__":
    main()
