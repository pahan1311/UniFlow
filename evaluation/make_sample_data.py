"""
make_sample_data.py
-------------------
Generates a SYNTHETIC labelled dataset for smoke-testing the evaluation
pipeline.

  *** DO NOT REPORT RESULTS OBTAINED ON THIS DATA. ***

It exists so you can run the full pipeline end to end -- baselines,
calibration, selective routing, OOD, benchmark -- and confirm every
stage executes and every figure renders, before your real corpus is
ready. Scores computed here are meaningless: the text is generated from
a handful of templates, so it is far more separable than genuine student
mail, and every model will look excellent.

Replace data/sample_inquiries.csv with your real data/inquiries.csv as
soon as you have it, and delete the sample file so it cannot be confused
with the real corpus.

Usage:
    python evaluation/make_sample_data.py --n 600
"""

import argparse
import random
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT = BASE_DIR / "data" / "sample_inquiries.csv"
SEED = 42

TEMPLATES = {
    "Academic Documents": [
        "I need an official {doc} sent to {place} for my {purpose}",
        "How do I request a certified copy of my {doc}",
        "My {doc} has an incorrect spelling of my name, how is it amended",
        "Requesting a {doc} for visa purposes, what is the processing time",
        "Can I collect my {doc} in person from the office this week",
    ],
    "Admissions": [
        "What are the entry requirements for the {programme} programme",
        "I submitted my application for {programme} but have not had a response",
        "Is the deadline for {programme} applications still open",
        "Can I transfer credits from another university into {programme}",
        "How many places are available on the {programme} course this intake",
    ],
    "Examinations": [
        "When are the {sem} semester examination results released",
        "I was ill during the {sem} exam and need to apply for a repeat",
        "There is an error in my {sem} examination grade for one module",
        "How do I apply to re-sit the {sem} semester paper",
        "Where is the examination hall for the {sem} semester paper",
    ],
    "Fees and Payments": [
        "My tuition payment of {amount} was deducted twice from my account",
        "I have not received a receipt for the {amount} I paid last week",
        "Can I pay the {amount} balance in instalments this semester",
        "The portal still shows an outstanding balance of {amount} after payment",
        "What is the deadline for settling the {amount} registration fee",
    ],
    "General Administration": [
        "How do I update my {detail} on the student record system",
        "I need a letter confirming my student status for {purpose}",
        "Who should I contact about {detail} for the student union",
        "Can you confirm the office opening hours during the vacation",
        "I would like to formally withdraw and need to know the procedure",
    ],
    "Learning Platform / IT Access": [
        "I cannot log in to the learning platform, it rejects my password",
        "The {system} portal shows an error whenever I try to upload work",
        "My university email account has been locked and I cannot reset it",
        "I have no access to the {system} module page for this semester",
        "The {system} system times out every time I open my dashboard",
    ],
    "Timetable and Scheduling": [
        "There is a clash between two lectures in my {sem} timetable",
        "The {sem} timetable has not been published for my programme yet",
        "My tutorial group was changed without notice in the {sem} schedule",
        "Which room is the {sem} semester lecture being held in",
        "Can I move to a different tutorial slot in the {sem} timetable",
    ],
}

URGENCY_SUFFIX = {
    "High": [
        ", this is urgent as the deadline is today",
        ", I need this resolved immediately, my exam is tomorrow",
        ", please treat as urgent, I am completely blocked",
        " and it must be sorted out today, there is no workaround",
    ],
    "Medium": [
        ", I would appreciate a reply within the next few days",
        ", the deadline is at the end of this week",
        ", please advise before the end of the week",
        " and I need to plan around it soon",
    ],
    "Low": [
        ", there is no rush, whenever convenient",
        ", just for future planning",
        ", this is a general question for next semester",
        ", no particular deadline on my side",
    ],
}

FILLERS = {
    "doc": ["transcript", "degree certificate", "academic record", "letter of completion"],
    "place": ["an overseas university", "a prospective employer", "the embassy"],
    "purpose": ["postgraduate application", "visa application", "job application"],
    "programme": ["Computer Science", "Business Administration", "Civil Engineering",
                  "Nursing", "Data Science"],
    "sem": ["first", "second", "third", "final"],
    "amount": ["25,000", "48,500", "12,000", "75,000"],
    "detail": ["my postal address", "my contact number", "my emergency contact"],
    "system": ["Moodle", "the VLE", "the student portal"],
}


def fill(template: str, rng: random.Random) -> str:
    text = template
    for key, options in FILLERS.items():
        token = "{" + key + "}"
        while token in text:
            text = text.replace(token, rng.choice(options), 1)
    return text


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic sample data")
    parser.add_argument("--n", type=int, default=600)
    parser.add_argument("--out", default=str(OUTPUT))
    args = parser.parse_args()

    rng = random.Random(SEED)
    categories = list(TEMPLATES)
    # Deliberately imbalanced, because a perfectly balanced synthetic set
    # would hide exactly the macro-F1 behaviour the pipeline exists to expose.
    urgency_weights = {"High": 0.22, "Medium": 0.45, "Low": 0.33}

    rows = []
    for _ in range(args.n):
        category = rng.choice(categories)
        urgency = rng.choices(list(urgency_weights), weights=list(
            urgency_weights.values()))[0]
        text = fill(rng.choice(TEMPLATES[category]), rng)
        text = text + rng.choice(URGENCY_SUFFIX[urgency])
        rows.append({"text": text, "category": category, "urgency": urgency})

    df = pd.DataFrame(rows).drop_duplicates(subset=["text"]).reset_index(drop=True)

    # Emit an explicit split column so the pipeline exercises the
    # "split supplied" path rather than the leakage-warning path.
    rng2 = random.Random(SEED + 1)
    splits = []
    for _ in range(len(df)):
        draw = rng2.random()
        splits.append("test" if draw < 0.15 else ("val" if draw < 0.30 else "train"))
    df["split"] = splits

    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output, index=False)

    print(f"Wrote {len(df)} synthetic rows to {output}")
    print(df["split"].value_counts().to_string())
    print("\n*** SYNTHETIC DATA — for pipeline testing only. Never report "
          "scores measured on it. ***")


if __name__ == "__main__":
    main()
