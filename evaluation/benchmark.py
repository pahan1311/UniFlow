"""
benchmark.py
------------
Inference cost measurement: parameter count, on-disk size and latency.

This is the other half of the multi-task experiment. Accuracy alone
cannot decide between two separate models and one shared encoder,
because the shared model's entire argument is that it costs roughly half
as much to run. Reporting a latency and size table next to the accuracy
table is what makes the architectural comparison a real trade-off
analysis rather than a leaderboard entry.

Latency is measured per single inquiry, since that is how the deployed
system runs -- one message at a time as mail arrives, not in batches.
Warm-up iterations are discarded because the first forward pass through
a freshly loaded PyTorch model is dominated by lazy allocation and is
not representative.

Reported statistics are median and p95 rather than the mean: response
latency is right-skewed, and the p95 is what an officer actually
experiences on a bad request.
"""

import argparse
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dataset import RESULTS_DIR, get_splits  # noqa: E402
from policy_files import MODELS_DIR, get_calibration, get_thresholds  # noqa: E402
from uniflow.ml.inference import classify_inquiry, load_all_models  # noqa: E402

SAMPLE_TEXTS = [
    "I cannot log in to the learning platform and my assignment is due tonight.",
    "Could you tell me when the provisional results for semester two are released?",
    "I need an official transcript posted to an overseas university for admission.",
    "There is a clash between two lectures in my timetable for the third week.",
    "My tuition payment was deducted twice from my account last Friday morning.",
]


def directory_size_mb(path: Path) -> float:
    if not path.exists():
        return 0.0
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / (1024 ** 2)


def count_parameters(models: dict) -> int:
    if models["backend"] == "multitask":
        model, _ = models["shared"]
        return sum(p.numel() for p in model.parameters())
    total = 0
    for task in ("category", "urgency"):
        model = models[task][0]
        total += sum(p.numel() for p in model.parameters())
    return total


def measure_latency(models, texts, iterations=40, warmup=5):
    """Per-inquiry end-to-end latency, including tokenisation and routing."""
    thresholds = get_thresholds()
    calibration = get_calibration()

    for i in range(warmup):
        classify_inquiry(texts[i % len(texts)], models, thresholds, calibration)

    timings = []
    for i in range(iterations):
        start = time.perf_counter()
        classify_inquiry(texts[i % len(texts)], models, thresholds, calibration)
        timings.append((time.perf_counter() - start) * 1000)

    array = np.array(timings)
    return {
        "median_ms": float(np.median(array)),
        "mean_ms": float(array.mean()),
        "p95_ms": float(np.quantile(array, 0.95)),
        "min_ms": float(array.min()),
        "max_ms": float(array.max()),
        "iterations": iterations,
    }


def main():
    parser = argparse.ArgumentParser(description="Inference cost benchmark")
    parser.add_argument("--iterations", type=int, default=40)
    parser.add_argument("--backend", choices=["auto", "dual", "multitask"],
                        default="auto",
                        help="force a backend; 'dual' ignores models/multitask/")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    try:
        _, _, test, _ = get_splits(quiet=True)
        texts = test["text"].head(20).tolist() or SAMPLE_TEXTS
    except SystemExit:
        texts = SAMPLE_TEXTS
        print("  (no dataset found; benchmarking on built-in sample inquiries)")

    prefer_multitask = args.backend != "dual"
    models = load_all_models(prefer_multitask=prefer_multitask)
    backend = models["backend"]
    print(f"\nBenchmarking backend: {backend}")

    if backend == "multitask":
        size = directory_size_mb(MODELS_DIR / "multitask")
        forward_passes = 1
    else:
        size = (directory_size_mb(MODELS_DIR / "distilbert_category")
                + directory_size_mb(MODELS_DIR / "distilbert_urgency"))
        forward_passes = 2

    parameters = count_parameters(models)
    latency = measure_latency(models, texts, args.iterations)

    record = {
        "backend": backend,
        "parameters": parameters,
        "parameters_millions": round(parameters / 1e6, 2),
        "disk_mb": round(size, 1),
        "forward_passes_per_inquiry": forward_passes,
        "latency": latency,
        "device": "cpu" if not torch.cuda.is_available() else torch.cuda.get_device_name(0),
        "torch_threads": torch.get_num_threads(),
        "platform": platform.platform(),
        "python": platform.python_version(),
    }

    print(f"  parameters   {record['parameters_millions']:.2f}M")
    print(f"  disk         {record['disk_mb']:.1f} MB")
    print(f"  fwd passes   {forward_passes} per inquiry")
    print(f"  latency      median {latency['median_ms']:.1f} ms, "
          f"p95 {latency['p95_ms']:.1f} ms")
    print(f"  device       {record['device']} ({record['torch_threads']} threads)")

    output = RESULTS_DIR / f"benchmark_{backend}.json"
    with open(output, "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2)
    print(f"\nWrote {output}")

    other = RESULTS_DIR / f"benchmark_{'dual' if backend == 'multitask' else 'multitask'}.json"
    if other.exists():
        with open(other, "r", encoding="utf-8") as f:
            comparison = json.load(f)
        speedup = comparison["latency"]["median_ms"] / latency["median_ms"]
        size_ratio = comparison["disk_mb"] / max(record["disk_mb"], 1e-9)
        print(f"\nAgainst {comparison['backend']}: "
              f"{speedup:.2f}x latency, {size_ratio:.2f}x disk size")
        print("Put both rows in the architecture comparison table in your report.")


if __name__ == "__main__":
    main()
