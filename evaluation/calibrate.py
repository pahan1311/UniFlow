"""
calibrate.py
------------
Post-hoc confidence calibration by temperature scaling.

The problem: a fine-tuned transformer is systematically overconfident.
It will report 0.97 on inputs it gets wrong a fifth of the time. The
dashboard displays that number to a department officer as "routing
confidence", and the selective-routing threshold is defined in terms of
it, so an uncalibrated number makes both the interface and the
abstention policy dishonest.

The fix (Guo et al., 2017, "On Calibration of Modern Neural Networks"):
learn a single scalar T and report softmax(logits / T) instead of
softmax(logits). T is fitted by minimising negative log-likelihood on
the VALIDATION split -- never the test split, which stays untouched for
reporting. Because dividing every logit by the same constant cannot
change which one is largest, accuracy is provably unchanged; only the
confidence values move. That is what makes this a safe post-hoc step
rather than a reason to retrain.

Outputs:
  models/calibration.json          consumed by the live app
  results/calibration_metrics.json ECE / NLL / Brier before and after
  results/reliability_<task>.png   reliability diagrams for the report
"""

import json
import sys
from pathlib import Path

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dataset import RESULTS_DIR, get_splits  # noqa: E402
from logits_cache import cached_logits, labels_for  # noqa: E402
from policy_files import CALIBRATION_FILE  # noqa: E402

N_BINS = 15


def expected_calibration_error(confidences, correct, n_bins=N_BINS):
    """ECE with equal-width confidence bins.

    Bins predictions by confidence, and in each bin compares the mean
    confidence against the observed accuracy. ECE is the sample-weighted
    mean of those gaps: 0 is perfect calibration. Reported as a
    percentage in the thesis.
    """
    confidences = np.asarray(confidences, dtype=float)
    correct = np.asarray(correct, dtype=float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)

    error = 0.0
    rows = []
    for i in range(n_bins):
        low, high = edges[i], edges[i + 1]
        mask = (confidences > low) & (confidences <= high) if i > 0 else (
            confidences >= low) & (confidences <= high)
        count = int(mask.sum())
        if count == 0:
            rows.append({"bin_low": low, "bin_high": high, "count": 0,
                         "accuracy": None, "confidence": None})
            continue
        bin_accuracy = float(correct[mask].mean())
        bin_confidence = float(confidences[mask].mean())
        error += (count / len(confidences)) * abs(bin_accuracy - bin_confidence)
        rows.append({"bin_low": low, "bin_high": high, "count": count,
                     "accuracy": bin_accuracy, "confidence": bin_confidence})

    return float(error), rows


def negative_log_likelihood(logits, labels, temperature=1.0):
    tensor = torch.tensor(logits, dtype=torch.float32) / temperature
    target = torch.tensor(labels, dtype=torch.long)
    return float(torch.nn.functional.cross_entropy(tensor, target).item())


def brier_score(probs, labels):
    """Multiclass Brier score: mean squared error against the one-hot target."""
    probs = np.asarray(probs)
    one_hot = np.zeros_like(probs)
    one_hot[np.arange(len(labels)), labels] = 1.0
    return float(np.mean(np.sum((probs - one_hot) ** 2, axis=1)))


MIN_TEMPERATURE = 0.05
MAX_TEMPERATURE = 20.0


def fit_temperature(logits, labels, max_iterations=200):
    """Fits T by minimising NLL with L-BFGS.

    Optimising log(T) rather than T keeps the temperature strictly
    positive without needing a constrained optimiser.

    The result is clamped to [0.05, 20]. The clamp matters in one
    degenerate case: if the model classifies the validation split
    perfectly, NLL is minimised by T -> 0, because infinite confidence
    is never penalised when nothing is ever wrong. A temperature of
    ~0.0004 would then be written to calibration.json and every
    prediction in the live app would report 100% confidence, which is
    the opposite of the honesty this stage exists to provide. A
    temperature at the clamp boundary is a signal that the validation
    split is too small or too easy, and the caller warns accordingly.
    """
    logit_tensor = torch.tensor(logits, dtype=torch.float32)
    label_tensor = torch.tensor(labels, dtype=torch.long)
    log_temperature = torch.zeros(1, requires_grad=True)

    optimiser = torch.optim.LBFGS([log_temperature], lr=0.1,
                                  max_iter=max_iterations)
    loss_fn = torch.nn.CrossEntropyLoss()

    def closure():
        optimiser.zero_grad()
        loss = loss_fn(logit_tensor / torch.exp(log_temperature), label_tensor)
        loss.backward()
        return loss

    optimiser.step(closure)
    fitted = float(torch.exp(log_temperature).detach().item())

    if not np.isfinite(fitted):
        return 1.0
    return float(np.clip(fitted, MIN_TEMPERATURE, MAX_TEMPERATURE))


def plot_reliability(rows_before, rows_after, task, ece_before, ece_after, path):
    """Reliability diagram: bin accuracy against bin confidence."""
    fig, ax = plt.subplots(figsize=(5.6, 5.2))
    ax.plot([0, 1], [0, 1], "--", color="#9bb5a8", linewidth=1.2,
            label="perfect calibration")

    for rows, colour, marker, label, ece in (
        (rows_before, "#d68c1a", "o", "before", ece_before),
        (rows_after, "#16835f", "s", "after", ece_after),
    ):
        points = [(r["confidence"], r["accuracy"]) for r in rows if r["count"] > 0]
        if points:
            x, y = zip(*points)
            ax.plot(x, y, marker=marker, color=colour, linewidth=1.6,
                    markersize=5, label=f"{label} (ECE {ece:.3f})")

    ax.set_xlabel("Mean predicted confidence")
    ax.set_ylabel("Observed accuracy")
    ax.set_title(f"Reliability diagram — {task}")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.25)
    ax.legend(loc="upper left", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def calibrate_task(task, val_df, test_df):
    """Fits T on validation, reports the effect on the test split."""
    print(f"\n--- calibration: {task} ---")
    labels = labels_for(task)
    label_to_id = {label: i for i, label in enumerate(labels)}

    val_logits = cached_logits(val_df["text"], task, "val")
    test_logits = cached_logits(test_df["text"], task, "test")

    val_mask = val_df[task].isin(label_to_id)
    test_mask = test_df[task].isin(label_to_id)
    if not val_mask.all() or not test_mask.all():
        print(f"  ! dropping {int((~val_mask).sum() + (~test_mask).sum())} rows whose "
              f"label is outside the model's label set")

    val_logits = val_logits[val_mask.to_numpy()]
    test_logits = test_logits[test_mask.to_numpy()]
    val_labels = np.array([label_to_id[l] for l in val_df[task][val_mask]])
    test_labels = np.array([label_to_id[l] for l in test_df[task][test_mask]])

    if len(val_labels) < 20:
        print(f"  ! only {len(val_labels)} validation items; the fitted temperature "
              f"will be unstable. Report this as a limitation.")

    temperature = fit_temperature(val_logits, val_labels)
    print(f"  fitted T = {temperature:.4f} on {len(val_labels)} validation items")
    if temperature <= MIN_TEMPERATURE * 1.01 or temperature >= MAX_TEMPERATURE * 0.99:
        print("    ! T hit the clamp boundary. This usually means the model is "
              "perfect (or near-perfect) on the validation split, which leaves "
              "the temperature unidentifiable. Treat the calibration numbers as "
              "unreliable and say so in the report.")
    elif temperature > 1:
        print("    T > 1 means the model was overconfident and has been softened.")
    elif temperature < 1:
        print("    T < 1 means the model was underconfident and has been sharpened.")

    def summarise(temp):
        scaled = torch.tensor(test_logits, dtype=torch.float32) / temp
        probs = torch.softmax(scaled, dim=-1).numpy()
        predicted = probs.argmax(axis=1)
        confidences = probs.max(axis=1)
        correct = (predicted == test_labels).astype(float)
        ece, rows = expected_calibration_error(confidences, correct)
        return {
            "ece": ece,
            "nll": negative_log_likelihood(test_logits, test_labels, temp),
            "brier": brier_score(probs, test_labels),
            "accuracy": float(correct.mean()),
            "mean_confidence": float(confidences.mean()),
            "bins": rows,
        }

    before = summarise(1.0)
    after = summarise(temperature)

    print(f"  ECE   {before['ece']:.4f} -> {after['ece']:.4f}")
    print(f"  NLL   {before['nll']:.4f} -> {after['nll']:.4f}")
    print(f"  Brier {before['brier']:.4f} -> {after['brier']:.4f}")
    print(f"  mean confidence {before['mean_confidence']:.4f} -> "
          f"{after['mean_confidence']:.4f} (accuracy {after['accuracy']:.4f}, "
          f"unchanged by construction)")

    plot_reliability(before["bins"], after["bins"], task, before["ece"],
                     after["ece"], RESULTS_DIR / f"reliability_{task}.png")

    return temperature, {"before": before, "after": after,
                         "temperature": temperature,
                         "n_val": len(val_labels), "n_test": len(test_labels)}


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    train, val, test, _ = get_splits(csv_path)

    if val.empty:
        raise SystemExit(
            "Validation split is empty. Temperature must be fitted on data that is "
            "neither training nor test data; add a val split to your CSV."
        )

    calibration, metrics = {}, {}
    for task in ("category", "urgency"):
        temperature, task_metrics = calibrate_task(task, val, test)
        calibration[task] = {"temperature": temperature}
        metrics[task] = task_metrics

    with open(CALIBRATION_FILE, "w", encoding="utf-8") as f:
        json.dump(calibration, f, indent=2)
    with open(RESULTS_DIR / "calibration_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nWrote {CALIBRATION_FILE}")
    print("The dashboard now reports calibrated confidences automatically.")


if __name__ == "__main__":
    main()
