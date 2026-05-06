from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare baseline vs distillation run metrics.")
    parser.add_argument("--baseline-metrics", type=str, required=True)
    parser.add_argument("--distill-metrics", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="./outputs/compare")
    parser.add_argument("--label", type=str, default="resnet18")
    return parser.parse_args()


def load_metrics(path: str) -> dict:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def extract_last(history: list[dict], key: str) -> float:
    if not history:
        return 0.0
    return float(history[-1].get(key, 0.0))


def main() -> None:
    args = parse_args()
    baseline = load_metrics(args.baseline_metrics)
    distill = load_metrics(args.distill_metrics)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "model": args.label,
        "baseline_best_val_acc": float(baseline.get("best_val_acc", 0.0)),
        "distill_best_val_acc": float(distill.get("best_val_acc", 0.0)),
        "baseline_test_acc": float(baseline.get("test_acc", 0.0)),
        "distill_test_acc": float(distill.get("test_acc", 0.0)),
        "baseline_last_val_loss": extract_last(baseline.get("history", []), "val_loss"),
        "distill_last_val_total_loss": extract_last(distill.get("history", []), "val_total_loss"),
    }

    with (output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    fig, ax = plt.subplots(figsize=(6, 4))
    labels = ["baseline", "distill"]
    values = [summary["baseline_test_acc"], summary["distill_test_acc"]]
    ax.bar(labels, values)
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Test Accuracy")
    ax.set_title(f"Baseline vs Distillation ({args.label})")
    fig.tight_layout()
    fig.savefig(output_dir / "test_acc_compare.png", dpi=140)
    plt.close(fig)

    print(json.dumps(summary, indent=2))
    print(f"Saved: {output_dir / 'summary.json'}")
    print(f"Saved: {output_dir / 'test_acc_compare.png'}")


if __name__ == "__main__":
    main()
