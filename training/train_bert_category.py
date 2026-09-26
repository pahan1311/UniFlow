# -*- coding: utf-8 -*-
"""train_bert_category.ipynb

Companion to train_bert_urgency_text_only_fixed.py, adapted for the
7-way category task and switched to bert-base-uncased so both heads
of the "dual" backend are the same base model.

This notebook trains a BERT classifier using only `Student_Query` to
predict `University_Category`:

- Admissions
- Examinations
- Fees and Payments
- Timetable and Scheduling
- Academic Documents
- Learning Platform / IT Access
- General Administration

## Run order
Run every cell from top to bottom.

When you reach the dataset upload section, select the same three
files used for the urgency model:

1. `university_issue_train_text_only.csv`
2. `university_issue_validation_text_only.csv`
3. `university_issue_test_text_only.csv`

## 1. Install packages
"""

# Commented out IPython magic to ensure Python compatibility.
# %pip install -q "transformers==4.57.2" "datasets>=3.0,<5" "accelerate>=1.0,<2" "scikit-learn>=1.3" "pandas>=2.0" "matplotlib>=3.7" "ipywidgets>=8.0"

"""## 2. Imports, random seed, and device check"""

from pathlib import Path
import json
import random
import shutil
import warnings

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from IPython.display import display
from datasets import Dataset
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
)

warnings.filterwarnings("ignore")

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", DEVICE)
if DEVICE == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))
else:
    print("No GPU detected. Training will work but will be much slower.")

"""## 3. Browse / upload the dataset files

Same three CSVs as the urgency notebook -- both label columns live in
the same files, so nothing new needs to be exported.
"""

EXPECTED_DATASET_NAMES = [
    "university_issue_train_text_only.csv",
    "university_issue_validation_text_only.csv",
    "university_issue_test_text_only.csv",
]

IS_COLAB = False
uploader = None

try:
    from google.colab import files as colab_files
    IS_COLAB = True

    print("Select the 3 text-only dataset CSV files.")
    uploaded = colab_files.upload()

    for filename, content in uploaded.items():
        Path(filename).write_bytes(content)

    print("\nUploaded:")
    for filename in uploaded:
        print(" -", filename)

except ImportError:
    import ipywidgets as widgets

    print("Jupyter detected.")
    print("Click Browse below and choose the 3 text-only dataset CSV files.")
    print("After selecting them, run the NEXT cell.")

    uploader = widgets.FileUpload(
        accept=".csv",
        multiple=True,
        description="Browse CSV files"
    )
    display(uploader)

"""## 4. Save uploaded files, locate them, and create `train_df`, `val_df`, `test_df`"""

def save_jupyter_widget_uploads(widget, destination=Path.cwd()):
    if widget is None:
        return []

    value = widget.value
    if not value:
        return []

    saved = []

    if isinstance(value, (tuple, list)):
        items = value
    elif isinstance(value, dict):
        items = []
        for key, info in value.items():
            if isinstance(info, dict):
                item = dict(info)
                item.setdefault("name", key)
            else:
                item = {"name": key, "content": info}
            items.append(item)
    else:
        items = []

    for item in items:
        name = item.get("name") or item.get("metadata", {}).get("name")
        content = item.get("content")

        if name is None or content is None:
            continue

        if isinstance(content, memoryview):
            content = content.tobytes()
        else:
            content = bytes(content)

        path = destination / name
        path.write_bytes(content)
        saved.append(path)

    return saved

if not IS_COLAB and uploader is not None:
    saved_files = save_jupyter_widget_uploads(uploader)
    if saved_files:
        print("Saved uploaded files:")
        for p in saved_files:
            print(" -", p.resolve())
    else:
        print("No new widget uploads were found. I will also search existing notebook folders.")


SEARCH_ROOTS = []
for candidate in [
    Path.cwd(),
    Path("/content"),
    Path("/kaggle/working"),
    Path("/mnt/data"),
]:
    if candidate.exists() and candidate not in SEARCH_ROOTS:
        SEARCH_ROOTS.append(candidate)


def find_file_exact(filename):
    for root in SEARCH_ROOTS:
        p = root / filename
        if p.exists():
            return p
    for root in SEARCH_ROOTS:
        try:
            matches = list(root.rglob(filename))
            if matches:
                return matches[0]
        except (PermissionError, OSError):
            pass
    return None


def all_visible_csvs():
    found = []
    seen = set()
    for root in SEARCH_ROOTS:
        try:
            for p in root.glob("*.csv"):
                key = str(p.resolve())
                if key not in seen:
                    found.append(p)
                    seen.add(key)
        except (PermissionError, OSError):
            pass
    return found


def choose_by_role(role):
    csvs = all_visible_csvs()
    role = role.lower()

    def valid_name(p):
        name = p.name.lower()
        if role == "train":
            return "train" in name
        if role == "validation":
            return "validation" in name or "valid" in name or "val" in name
        if role == "test":
            return "test" in name
        return False

    candidates = [p for p in csvs if valid_name(p)]
    text_only = [p for p in candidates if "text_only" in p.name.lower() or "text-only" in p.name.lower()]
    if text_only:
        candidates = text_only

    return candidates[0] if candidates else None


train_path = find_file_exact("university_issue_train_text_only.csv") or choose_by_role("train")
val_path   = find_file_exact("university_issue_validation_text_only.csv") or choose_by_role("validation")
test_path  = find_file_exact("university_issue_test_text_only.csv") or choose_by_role("test")

print("\nResolved files:")
print("Train      :", train_path)
print("Validation :", val_path)
print("Test       :", test_path)

missing = []
if train_path is None:
    missing.append("training CSV")
if val_path is None:
    missing.append("validation CSV")
if test_path is None:
    missing.append("test CSV")

if missing:
    print("\nCSV files currently visible:")
    for p in all_visible_csvs():
        print(" -", p)
    raise FileNotFoundError(
        "Missing: " + ", ".join(missing) +
        ". Run the previous upload cell and select the *_text_only.csv files."
    )

train_df = pd.read_csv(train_path)
val_df   = pd.read_csv(val_path)
test_df  = pd.read_csv(test_path)

print("\nLoaded successfully:")
print("train_df:", train_df.shape)
print("val_df  :", val_df.shape)
print("test_df :", test_df.shape)

print("\nFirst training rows:")
display(train_df.head(3))

"""## 5. Validate the datasets before training"""

EXPECTED_COLUMNS = {"Student_Query", "University_Category"}
VALID_LABELS = {
    "Admissions",
    "Examinations",
    "Fees and Payments",
    "Timetable and Scheduling",
    "Academic Documents",
    "Learning Platform / IT Access",
    "General Administration",
}

for name, df in [
    ("train", train_df),
    ("validation", val_df),
    ("test", test_df),
]:
    missing_cols = EXPECTED_COLUMNS - set(df.columns)
    assert not missing_cols, f"{name} is missing columns: {missing_cols}"

    assert df["Student_Query"].notna().all(), f"Missing Student_Query values in {name}"
    assert df["University_Category"].notna().all(), f"Missing University_Category values in {name}"

    labels = set(df["University_Category"].astype(str).str.strip().unique())
    assert labels.issubset(VALID_LABELS), f"Unexpected labels in {name}: {labels - VALID_LABELS}"

    assert df["Student_Query"].astype(str).str.strip().ne("").all(), f"Blank queries in {name}"

print("All basic dataset checks passed.")

print("\nClass counts:")
for name, df in [("Train", train_df), ("Validation", val_df), ("Test", test_df)]:
    print(f"\n{name}")
    print(df["University_Category"].value_counts())

"""## 6. Check scenario-family leakage (if the column exists)

Same check as the urgency notebook -- confirms no Scenario_Family
appears in more than one split, which would let the model memorise a
phrasing template rather than generalise.
"""

if all("Scenario_Family" in df.columns for df in [train_df, val_df, test_df]):
    train_families = set(train_df["Scenario_Family"].dropna().astype(str))
    val_families   = set(val_df["Scenario_Family"].dropna().astype(str))
    test_families  = set(test_df["Scenario_Family"].dropna().astype(str))

    print("Train intersect Validation families:", len(train_families & val_families))
    print("Train intersect Test families      :", len(train_families & test_families))
    print("Validation intersect Test families :", len(val_families & test_families))
else:
    print("Scenario_Family is not present in all files, so this optional check was skipped.")

"""## 7. Normalize labels and inspect examples"""

for df in [train_df, val_df, test_df]:
    df["Student_Query"] = df["Student_Query"].astype(str).str.strip()
    df["University_Category"] = df["University_Category"].astype(str).str.strip()

CATEGORY_LABELS = sorted(VALID_LABELS)
LABEL2ID = {label: i for i, label in enumerate(CATEGORY_LABELS)}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}

print("LABEL2ID:", LABEL2ID)

for label in CATEGORY_LABELS:
    print(f"\n--- {label} examples ---")
    subset = train_df.loc[train_df["University_Category"] == label, "Student_Query"].head(2)
    for text in subset:
        print("-", text[:250])

"""## 8. Load the pretrained BERT tokenizer

bert-base-uncased, matching the urgency model, so both heads of the
"dual" backend are the same base architecture.
"""

MODEL_NAME = "bert-base-uncased"
MAX_LENGTH = 192

print("Loading tokenizer:", MODEL_NAME)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
print("Tokenizer loaded.")

"""## 9. Convert pandas DataFrames to Hugging Face datasets"""

def dataframe_to_hf_dataset(df):
    temp = df[["Student_Query", "University_Category"]].copy()
    temp["labels"] = temp["University_Category"].map(LABEL2ID).astype(int)

    ds = Dataset.from_pandas(
        temp[["Student_Query", "labels"]],
        preserve_index=False
    )

    def tokenize_batch(batch):
        return tokenizer(
            batch["Student_Query"],
            truncation=True,
            max_length=MAX_LENGTH,
        )

    ds = ds.map(tokenize_batch, batched=True)
    ds = ds.remove_columns(["Student_Query"])
    return ds

train_ds = dataframe_to_hf_dataset(train_df)
val_ds   = dataframe_to_hf_dataset(val_df)
test_ds  = dataframe_to_hf_dataset(test_df)

print(train_ds)
print(val_ds)
print(test_ds)

"""## 10. Build a fresh BERT classifier"""

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=len(CATEGORY_LABELS),
    label2id=LABEL2ID,
    id2label=ID2LABEL,
)

model.to(DEVICE)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
print("Fresh model created.")

"""## 11. Evaluation metrics"""

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)

    accuracy = accuracy_score(labels, preds)
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        labels,
        preds,
        average="macro",
        zero_division=0,
    )

    return {
        "accuracy": accuracy,
        "precision_macro": precision_macro,
        "recall_macro": recall_macro,
        "f1_macro": f1_macro,
    }

"""## 12. Training configuration

Same hyperparameters as the urgency notebook, so the two heads are
trained under a directly comparable protocol.
"""

OUTPUT_DIR = "bert_category_training_output"

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    num_train_epochs=4,
    weight_decay=0.01,
    warmup_ratio=0.1,

    eval_strategy="epoch",
    save_strategy="epoch",
    logging_strategy="steps",
    logging_steps=50,

    load_best_model_at_end=True,
    metric_for_best_model="f1_macro",
    greater_is_better=True,
    save_total_limit=2,

    fp16=torch.cuda.is_available(),
    report_to="none",
    seed=SEED,
    data_seed=SEED,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    processing_class=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
)

print("Trainer ready.")

"""## 13. Train BERT"""

train_result = trainer.train()
print("\nTraining complete.")
print(train_result)

"""## 14. Evaluate on the validation set"""

val_metrics = trainer.evaluate(val_ds)
print("Validation metrics:")
for key, value in val_metrics.items():
    if isinstance(value, (int, float)):
        print(f"{key}: {value:.4f}")
    else:
        print(f"{key}: {value}")

"""## 15. Final evaluation on the held-out test set"""

test_output = trainer.predict(test_ds)
test_logits = test_output.predictions
test_true_ids = test_output.label_ids
test_pred_ids = np.argmax(test_logits, axis=-1)

print("Test metrics:")
for key, value in test_output.metrics.items():
    if isinstance(value, (int, float)):
        print(f"{key}: {value:.4f}")
    else:
        print(f"{key}: {value}")

print("\nClassification report:\n")
print(classification_report(
    test_true_ids,
    test_pred_ids,
    labels=list(range(len(CATEGORY_LABELS))),
    target_names=CATEGORY_LABELS,
    digits=4,
    zero_division=0,
))

"""## 16. Confusion matrix"""

cm = confusion_matrix(test_true_ids, test_pred_ids, labels=list(range(len(CATEGORY_LABELS))))

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=CATEGORY_LABELS,
)

fig, ax = plt.subplots(figsize=(9, 9))
disp.plot(values_format="d", ax=ax, xticks_rotation=45)
plt.title("BERT Category Classifier - Test Confusion Matrix")
plt.tight_layout()
plt.show()

"""## 17. Reusable prediction function"""

def predict_category(text):
    model.eval()

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_LENGTH,
    )
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        logits = model(**inputs).logits
        probabilities = torch.softmax(logits, dim=-1)[0].cpu().numpy()

    predicted_id = int(np.argmax(probabilities))

    return {
        "prediction": ID2LABEL[predicted_id],
        **{label: float(probabilities[LABEL2ID[label]]) for label in CATEGORY_LABELS},
    }

print("Prediction function ready.")

"""## 18. Save the final trained model

Saved under the same folder name uniflow/ml/inference.py already
expects (distilbert_category) -- the loader reads whatever model is
actually inside via AutoModelForSequenceClassification, so the folder
name does not need to match the architecture. Rename the destination
below if you would rather keep the old DistilBERT checkpoint alongside
this one instead of replacing it.
"""

FINAL_MODEL_DIR = Path("distilbert_category")  # matches models/distilbert_category on the server
FINAL_MODEL_DIR.mkdir(parents=True, exist_ok=True)

trainer.save_model(str(FINAL_MODEL_DIR))
tokenizer.save_pretrained(str(FINAL_MODEL_DIR))

with open(FINAL_MODEL_DIR / "label_classes.json", "w", encoding="utf-8") as f:
    json.dump(CATEGORY_LABELS, f, indent=2)

with open(FINAL_MODEL_DIR / "label_mapping.json", "w", encoding="utf-8") as f:
    json.dump(
        {
            "label2id": LABEL2ID,
            "id2label": {str(k): v for k, v in ID2LABEL.items()},
            "model_input": "Student_Query only",
            "base_model": MODEL_NAME,
            "max_length": MAX_LENGTH,
        },
        f,
        indent=2,
    )

print("Saved model to:", FINAL_MODEL_DIR.resolve())

"""## 19. Zip the trained model for download"""

zip_path = shutil.make_archive(
    base_name="distilbert_category",
    format="zip",
    root_dir=FINAL_MODEL_DIR,
)

print("Created:", zip_path)

try:
    from google.colab import files as colab_files
    colab_files.download(zip_path)
except ImportError:
    print("Not running in Colab. Download the ZIP from your Jupyter file browser:")
    print(zip_path)

"""# Finished

```text
Student_Query -> BERT -> one of the 7 university categories
```

After downloading, replace the contents of models/distilbert_category/
on your machine with this model's files (model.safetensors, config.json,
tokenizer files, label_classes.json / label_mapping.json), keeping the
folder name distilbert_category unchanged so uniflow keeps finding it.

Then re-run the evaluation pipeline so Chapter 06's numbers reflect the
new model:

    python evaluation/combine_dataset.py --train ... --val ... --test ...   (only if data/inquiries.csv is missing)
    python evaluation/run_all.py --skip dataset

--skip dataset avoids rebuilding data/inquiries.csv; every other stage
(baselines, evaluate, calibrate, selective, ood, benchmark) needs to
rerun because the category model changed. Delete
evaluation/results/logits_category_*.npy first if you keep the old
files around -- the cache is keyed by row count, not by model, so a
same-size dataset with a new model needs the cache cleared explicitly
(see logits_cache.py).
"""
