"""
selective.py
------------
Selective prediction: choosing the confidence threshold tau at which the
system stops auto-routing and asks a human instead.

The framing is that a misroute is not free. An inquiry sent to the wrong
department is read, re-read, forwarded and delayed -- often past its SLA
target. Abstaining costs something too, but far less: an item sits in
Triage until a human places it. So the design question is not "how
accurate is the classifier" but "at what confidence does auto-routing
stop being worth it".

The risk-coverage curve answers that. Sort the test items by confidence,
and for each possible tau compute:

    coverage = fraction of inquiries auto-routed (confidence >= tau)
    risk     = error rate among only those auto-routed items

Coverage falls and risk falls as tau rises. AURC (area under the curve)
summarises the confidence ranking's quality independently of any chosen
tau, and is the number to report when comparing calibrated against
uncalibrated scores, or one model against another.

tau itself is selected on the VALIDATION split as the smallest threshold
meeting a target risk (default 5% error among auto-routed items), then
that fixed tau is reported on the test split. Selecting tau on test data
and then reporting on the same data would be a textbook methodological
error, so the module refuses to do it.

Output: models/thresholds.json (consumed live by the app) plus the curve
data and figure for the report.
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

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dataset import RESULTS_DIR, get_splits  # noqa: E402
from logits_cache import cached_logits, labels_for  # noqa: E402
from policy_files import THRESHOLDS_FILE, get_calibration  # noqa: E402

DEFAULT_TARGET_RISK = 0.05


def confidences_and_correctness(logits, gold_labels, label_list, temperature):
    """Calibrated max-probability and a correctness flag per item."""
    label_to_id = {label: i for i, label in enumerate(label_list)}
    mask = np.array([label in label_to_id for label in gold_labels])
    logits = np.asarray(logits)[mask]
    targets = np.array([label_to_id[l] for l, keep in zip(gold_labels, mask) if keep])

    probs = torch.softmax(
        torch.tensor(logits, dtype=torch.float32) / max(temperature, 1e-6), dim=-1
    ).numpy()
    predicted = probs.argmax(axis=1)
    return probs.max(axis=1), (predicted == targets).astype(float)


def risk_coverage_curve(confidences, correct):
    """Risk and coverage at every distinct confidence threshold.

    Implemented by sorting once descending and taking cumulative means,
    so the whole curve costs O(n log n) rather than O(n) thresholds each
    scanning O(n) items.
    """
    order = np.argsort(-confidences)
    sorted_confidence = confidences[order]
    sorted_correct = correct[order]

    counts = np.arange(1, len(sorted_correct) + 1)
    cumulative_accuracy = np.cumsum(sorted_correct) / counts

    return pd.DataFrame({
        "tau": sorted_confidence,
        "coverage": counts / len(sorted_correct),
        "risk": 1.0 - cumulative_accuracy,
        "n_covered": counts,
    })


def area_under_risk_coverage(curve: pd.DataFrame) -> float:
    """AURC: mean risk across coverage levels. Lower is better."""
    return float(np.trapezoid(curve["risk"], curve["coverage"]) /
                 max(curve["coverage"].iloc[-1] - curve["coverage"].iloc[0], 1e-9))


def choose_tau(curve: pd.DataFrame, target_risk: float, min_coverage: float = 0.3):
    """Lowest tau whose selective risk is within target, keeping coverage usable.

    Preferring the *lowest* qualifying tau maximises how many inquiries
    are handled automatically, which is the point of the system; the
    min_coverage floor stops a degenerate solution that only routes a
    handful of trivially easy items.
    """
    eligible = curve[(curve["risk"] <= target_risk) &
                     (curve["coverage"] >= min_coverage)]
    if eligible.empty:
        fallback = curve[curve["coverage"] >= min_coverage]
        if fallback.empty:
            return float(curve["tau"].median()), False
        best = fallback.loc[fallback["risk"].idxmin()]
        return float(best["tau"]), False
    best = eligible.loc[eligible["coverage"].idxmax()]
    return float(best["tau"]), True


def evaluate_at_tau(confidences, correct, tau) -> dict:
    """What the operating point actually delivers."""
    routed = confidences >= tau
    n = len(confidences)
    if routed.sum() == 0:
        return {"tau": float(tau), "coverage": 0.0, "risk": None,
                "n_auto_routed": 0, "n_triaged": int(n),
                "errors_prevented": int((1 - correct).sum())}

    auto_errors = int((1 - correct[routed]).sum())
    total_errors = int((1 - correct).sum())
    return {
        "tau": float(tau),
        "coverage": float(routed.mean()),
        "risk": float(1 - correct[routed].mean()),
        "accuracy_on_routed": float(correct[routed].mean()),
        "n_auto_routed": int(routed.sum()),
        "n_triaged": int(n - routed.sum()),
        "errors_among_routed": auto_errors,
        "errors_prevented": total_errors - auto_errors,
        "full_coverage_risk": float(1 - correct.mean()),
    }


def plot_curve(curves: dict, taus: dict, task: str, path: Path):
    fig, ax = plt.subplots(figsize=(6.2, 4.6))
    colours = {"val": "#d68c1a", "test": "#16835f"}
    for split, curve in curves.items():
        ax.plot(curve["coverage"], curve["risk"], color=colours.get(split, "#555"),
                linewidth=1.8, label=f"{split} (AURC {area_under_risk_coverage(curve):.4f})")

    if "test" in taus and taus["test"] is not None:
        point = taus["test"]
        ax.scatter([point["coverage"]], [point["risk"]], color="#0e4936", zorder=5, s=55)
        ax.annotate(
            f"τ={point['tau']:.2f}\ncoverage {point['coverage']:.0%}, "
            f"risk {point['risk']:.1%}",
            (point["coverage"], point["risk"]),
            textcoords="offset points", xytext=(-14, 20), fontsize=8,
            bbox={"boxstyle": "round,pad=0.35", "fc": "#e8f7ef", "ec": "#16835f"},
        )

    ax.set_xlabel("Coverage — fraction auto-routed")
    ax.set_ylabel("Selective risk — error rate among auto-routed")
    ax.set_title(f"Risk–coverage curve — {task}")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def run_task(task, val_df, test_df, target_risk, calibration):
    print(f"\n--- selective prediction: {task} ---")
    labels = labels_for(task)
    temperature = calibration.get(task, {}).get("temperature", 1.0)
    if temperature == 1.0:
        print("  ! temperature is 1.0 — run calibrate.py first, otherwise tau is "
              "chosen against uncalibrated confidences.")

    val_logits = cached_logits(val_df["text"], task, "val")
    test_logits = cached_logits(test_df["text"], task, "test")

    val_conf, val_correct = confidences_and_correctness(
        val_logits, val_df[task].tolist(), labels, temperature)
    test_conf, test_correct = confidences_and_correctness(
        test_logits, test_df[task].tolist(), labels, temperature)

    val_curve = risk_coverage_curve(val_conf, val_correct)
    test_curve = risk_coverage_curve(test_conf, test_correct)

    tau, met_target = choose_tau(val_curve, target_risk)
    if met_target:
        print(f"  tau = {tau:.4f} chosen on validation for target risk "
              f"{target_risk:.0%}")
    else:
        print(f"  tau = {tau:.4f} — target risk {target_risk:.0%} was NOT achievable "
              f"at usable coverage; report this honestly rather than lowering the bar.")

    val_point = evaluate_at_tau(val_conf, val_correct, tau)
    test_point = evaluate_at_tau(test_conf, test_correct, tau)

    print(f"  test @ tau: coverage {test_point['coverage']:.1%}, "
          f"selective risk {test_point['risk']:.2%} "
          f"(vs {test_point['full_coverage_risk']:.2%} if everything were auto-routed)")
    print(f"  {test_point['n_triaged']} of {len(test_conf)} items sent to Triage; "
          f"{test_point['errors_prevented']} misroutes avoided")
    print(f"  AURC val {area_under_risk_coverage(val_curve):.4f}  "
          f"test {area_under_risk_coverage(test_curve):.4f}")

    test_curve.to_csv(RESULTS_DIR / f"risk_coverage_{task}.csv", index=False)
    plot_curve({"val": val_curve, "test": test_curve},
               {"test": test_point}, task,
               RESULTS_DIR / f"risk_coverage_{task}.png")

    return tau, {
        "target_risk": target_risk,
        "target_met": met_target,
        "temperature": temperature,
        "validation_operating_point": val_point,
        "test_operating_point": test_point,
        "aurc_val": area_under_risk_coverage(val_curve),
        "aurc_test": area_under_risk_coverage(test_curve),
    }


def main():
    parser = argparse.ArgumentParser(description="Selective prediction threshold search")
    parser.add_argument("csv", nargs="?", default=None)
    parser.add_argument("--target-risk", type=float, default=DEFAULT_TARGET_RISK,
                        help="acceptable error rate among auto-routed inquiries")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    train, val, test, _ = get_splits(Path(args.csv) if args.csv else None)
    if val.empty or test.empty:
        raise SystemExit("Both a validation and a test split are required.")

    calibration = get_calibration()

    existing = {}
    if THRESHOLDS_FILE.exists():
        with open(THRESHOLDS_FILE, "r", encoding="utf-8") as f:
            existing = json.load(f)

    metrics = {}
    for task in ("category", "urgency"):
        tau, task_metrics = run_task(task, val, test, args.target_risk, calibration)
        existing.setdefault(task, {})["tau"] = tau
        metrics[task] = task_metrics

    with open(THRESHOLDS_FILE, "w", encoding="utf-8") as f:
        json.dump(existing, f, indent=2)
    with open(RESULTS_DIR / "selective_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nWrote {THRESHOLDS_FILE}")
    print("The dashboard will now abstain to Triage below these thresholds.")


if __name__ == "__main__":
    main()
