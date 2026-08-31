"""
ood.py
------
Out-of-distribution detection: recognising inquiries that belong to none
of the seven departments.

A softmax classifier is a closed-set model. Given spam, a message in the
wrong language, or a question about something the university does not
handle, it must still output one of its seven labels, often with high
confidence. In a system that routes automatically that is a real failure
mode, not a curiosity: junk lands in a department queue and consumes an
officer's attention.

The detector used here is the free-energy score (Liu et al., 2020):

    E(x) = -logsumexp(logits)

In-distribution inputs excite at least one class strongly and so have
low energy; unfamiliar inputs produce uniformly small logits and higher
energy. Energy uses the absolute magnitude of the logits, whereas
max-softmax normalises it away -- which is why energy separates the
confidently-wrong cases that max-softmax misses.

Evaluation treats this as a binary detection problem over in-distribution
test items versus an out-of-scope probe set, reporting AUROC and
FPR@95%TPR. The threshold written to models/thresholds.json is set on
the VALIDATION split at 95% true-positive rate, i.e. it accepts 95% of
genuine inquiries by construction, and the cost of the detector is the
5% of real inquiries that get a Triage detour.

The probe set comes from data/ood_samples.csv if present (one column,
`text`). Otherwise a small built-in synthetic probe set is used, which
must be described in the report as synthetic -- it demonstrates the
mechanism but is weaker evidence than genuine out-of-scope mail
collected from the real inbox.
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from sklearn.metrics import roc_auc_score, roc_curve  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dataset import DATA_DIR, RESULTS_DIR, get_splits  # noqa: E402
from logits_cache import cached_logits  # noqa: E402
from policy_files import THRESHOLDS_FILE  # noqa: E402

OOD_CSV = DATA_DIR / "ood_samples.csv"
TARGET_TPR = 0.95

SYNTHETIC_OOD = [
    "Congratulations! You have won a free iPhone, click this link to claim your prize.",
    "Hi, are we still meeting for lunch at the usual place on Saturday afternoon?",
    "Please find attached the quarterly sales projection for the northern region.",
    "asdkjh asd kjahsd lkajshd lkajshdlkajshd",
    "What is the weather forecast for tomorrow morning in the city centre?",
    "I would like to order two large pepperoni pizzas for delivery this evening.",
    "URGENT: your bank account has been suspended, verify your details immediately.",
    "Can you recommend a good mechanic who can service a diesel engine nearby?",
    "The football match was postponed because of the heavy rain last night.",
    "Subscribe now for unlimited streaming of thousands of movies and shows.",
    "My laptop battery drains very fast, should I replace it with a new one?",
    "Looking to rent a two bedroom apartment close to the beach from next month.",
    "Reminder: your dental appointment is scheduled for the fifth of next month.",
    "Buy cheap watches online with worldwide shipping and huge discounts today.",
    "Could you send me the recipe for the chocolate cake you made last weekend?",
    "The train service between the two cities will be suspended for maintenance.",
    "I am writing to enquire about the price of bulk cement delivered to a site.",
    "Happy birthday! Hope you have a wonderful day with your family and friends.",
    "Please unsubscribe me from this mailing list, I no longer wish to receive it.",
    "test test test test test",
]


def load_probe_set() -> tuple[list[str], bool]:
    """Returns (texts, is_synthetic)."""
    if OOD_CSV.exists():
        df = pd.read_csv(OOD_CSV)
        column = next((c for c in df.columns if c.lower() in
                       {"text", "message", "query", "body"}), df.columns[0])
        texts = df[column].astype(str).str.strip()
        texts = [t for t in texts if t]
        if texts:
            return texts, False
    return list(SYNTHETIC_OOD), True


def energies(texts, task, split_name, cache=True) -> np.ndarray:
    """Energy scores for a split, from cached logits where available."""
    logits = cached_logits(texts, task, split_name, cache=cache)
    # E(x) = -logsumexp(logits), computed here rather than imported from
    # model_utils so this stage never has to import the model layer when
    # the logits are already on disk.
    tensor = torch.tensor(logits, dtype=torch.float32)
    return (-torch.logsumexp(tensor, dim=-1)).numpy()


def fpr_at_tpr(id_scores, ood_scores, target_tpr=TARGET_TPR):
    """False-positive rate when the detector accepts target_tpr of real inquiries.

    Labels: 1 = in-distribution (should be accepted). Because *low*
    energy means in-distribution, the score fed to the ROC is negated.
    """
    labels = np.concatenate([np.ones(len(id_scores)), np.zeros(len(ood_scores))])
    scores = np.concatenate([-id_scores, -ood_scores])
    fpr, tpr, thresholds = roc_curve(labels, scores)
    index = int(np.argmin(np.abs(tpr - target_tpr)))
    return float(fpr[index]), float(-thresholds[index]), float(tpr[index])


def plot_distributions(id_scores, ood_scores, threshold, auroc, path):
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    bins = np.linspace(
        min(id_scores.min(), ood_scores.min()),
        max(id_scores.max(), ood_scores.max()), 34,
    )
    ax.hist(id_scores, bins=bins, alpha=0.7, color="#16835f", label="in-distribution")
    ax.hist(ood_scores, bins=bins, alpha=0.7, color="#d68c1a", label="out-of-scope")
    ax.axvline(threshold, color="#0e4936", linestyle="--", linewidth=1.6,
               label=f"threshold {threshold:.2f}")
    ax.set_xlabel("Energy score  (lower = more familiar)")
    ax.set_ylabel("Count")
    ax.set_title(f"Energy separation — AUROC {auroc:.3f}")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Energy-based OOD detection")
    parser.add_argument("csv", nargs="?", default=None)
    parser.add_argument("--target-tpr", type=float, default=TARGET_TPR)
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    train, val, test, _ = get_splits(Path(args.csv) if args.csv else None)
    if val.empty or test.empty:
        raise SystemExit("Both a validation and a test split are required.")

    probe_texts, synthetic = load_probe_set()
    print(f"\nOOD probe set: {len(probe_texts)} items "
          f"({'synthetic built-in' if synthetic else OOD_CSV.name})")
    if synthetic:
        print("  ! This is a synthetic probe set. It demonstrates the mechanism but "
              "should be replaced with real out-of-scope mail before you claim a "
              "deployment-grade result.")

    task = "category"
    val_energy = energies(val["text"].tolist(), task, "val")
    test_energy = energies(test["text"].tolist(), task, "test")
    ood_energy = energies(probe_texts, task, "ood")

    labels = np.concatenate([np.ones(len(test_energy)), np.zeros(len(ood_energy))])
    scores = np.concatenate([-test_energy, -ood_energy])
    auroc = float(roc_auc_score(labels, scores))

    # The operating threshold is set on validation so the test-set AUROC
    # remains an untouched held-out number.
    threshold = float(np.quantile(val_energy, args.target_tpr))
    false_positive_rate, _, achieved_tpr = fpr_at_tpr(test_energy, ood_energy,
                                                      args.target_tpr)

    detected = float((ood_energy > threshold).mean())
    real_diverted = float((test_energy > threshold).mean())

    print(f"\n--- energy OOD detection ({task}) ---")
    print(f"  AUROC                         {auroc:.4f}")
    print(f"  FPR @ {args.target_tpr:.0%} TPR                {false_positive_rate:.4f}")
    print(f"  threshold (val {args.target_tpr:.0%} quantile) {threshold:.4f}")
    print(f"  out-of-scope items caught      {detected:.1%}")
    print(f"  real inquiries sent to Triage  {real_diverted:.1%}  "
          f"(the cost of the detector)")
    print(f"  mean energy: in-dist {test_energy.mean():.3f}, "
          f"out-of-scope {ood_energy.mean():.3f}")

    plot_distributions(test_energy, ood_energy, threshold, auroc,
                       RESULTS_DIR / "ood_energy.png")

    existing = {}
    if THRESHOLDS_FILE.exists():
        with open(THRESHOLDS_FILE, "r", encoding="utf-8") as f:
            existing = json.load(f)
    existing.setdefault("category", {})["energy_threshold"] = threshold
    with open(THRESHOLDS_FILE, "w", encoding="utf-8") as f:
        json.dump(existing, f, indent=2)

    with open(RESULTS_DIR / "ood_metrics.json", "w", encoding="utf-8") as f:
        json.dump({
            "auroc": auroc,
            "fpr_at_tpr": false_positive_rate,
            "achieved_tpr": achieved_tpr,
            "target_tpr": args.target_tpr,
            "threshold": threshold,
            "ood_detection_rate": detected,
            "in_distribution_diverted": real_diverted,
            "probe_set_synthetic": synthetic,
            "n_probe": len(probe_texts),
            "mean_energy_id": float(test_energy.mean()),
            "mean_energy_ood": float(ood_energy.mean()),
        }, f, indent=2)

    print(f"\nWrote {THRESHOLDS_FILE} and {RESULTS_DIR / 'ood_metrics.json'}")


if __name__ == "__main__":
    main()
