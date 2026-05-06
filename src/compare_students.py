from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate baseline/distill metrics across multiple student models."
    )
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        help="Format: model_label:baseline_metrics_path:distill_metrics_path",
    )
    parser.add_argument("--output-dir", type=str, default="./outputs/compare_students")
    return parser.parse_args()


def load_json(path: str) -> dict:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def parse_run_spec(spec: str) -> tuple[str, str, str]:
    parts = spec.split(":")
    if len(parts) != 3:
        raise ValueError(f"Invalid --run spec: {spec}")
    return parts[0], parts[1], parts[2]


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for spec in args.run:
        label, baseline_path, distill_path = parse_run_spec(spec)
        baseline = load_json(baseline_path)
        distill = load_json(distill_path)

        param_count = int(
            baseline.get(
                "model_total_params",
                distill.get("student_total_params", 0),
            )
        )
        row = {
            "model": label,
            "param_count": param_count,
            "baseline_val_acc": float(baseline.get("best_val_acc", 0.0)),
            "baseline_test_acc": float(baseline.get("test_acc", 0.0)),
            "baseline_test_f1": float(baseline.get("test_f1", 0.0)),
            "distill_val_acc": float(distill.get("best_val_acc", 0.0)),
            "distill_test_acc": float(distill.get("test_acc", 0.0)),
            "distill_test_f1": float(distill.get("test_f1", 0.0)),
        }
        rows.append(row)

    rows.sort(key=lambda x: x["param_count"])

    with (output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)

    with (output_dir / "summary.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "model",
                "param_count",
                "baseline_val_acc",
                "baseline_test_acc",
                "baseline_test_f1",
                "distill_val_acc",
                "distill_test_acc",
                "distill_test_f1",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    x_model = [r["model"] for r in rows]
    baseline_acc = [r["baseline_test_acc"] for r in rows]
    distill_acc = [r["distill_test_acc"] for r in rows]
    baseline_f1 = [r["baseline_test_f1"] for r in rows]
    distill_f1 = [r["distill_test_f1"] for r in rows]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    x = np.arange(len(x_model))
    width = 0.36

    axes[0].bar(x - width / 2, baseline_acc, width=width, label="baseline", color="#4C78A8")
    axes[0].bar(x + width / 2, distill_acc, width=width, label="distillation", color="#F58518")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(x_model)
    axes[0].set_xlabel("Student Model")
    axes[0].set_ylabel("Test Accuracy")
    axes[0].set_ylim(0.0, 1.0)
    axes[0].set_title("Baseline vs Distillation Accuracy")
    axes[0].legend()

    axes[1].bar(x - width / 2, baseline_f1, width=width, label="baseline", color="#4C78A8")
    axes[1].bar(x + width / 2, distill_f1, width=width, label="distillation", color="#F58518")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(x_model)
    axes[1].set_xlabel("Student Model")
    axes[1].set_ylabel("Test Macro-F1")
    axes[1].set_ylim(0.0, 1.0)
    axes[1].set_title("Baseline vs Distillation Macro-F1")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(output_dir / "acc_by_model.png", dpi=140)
    plt.close(fig)

    x_params = [r["param_count"] for r in rows]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].scatter(x_params, baseline_acc, label="baseline", color="#4C78A8", s=70)
    axes[0].scatter(x_params, distill_acc, label="distillation", color="#F58518", s=70)
    for i, model_name in enumerate(x_model):
        axes[0].annotate(model_name, (x_params[i], baseline_acc[i]), fontsize=8, xytext=(4, 4), textcoords="offset points")
        axes[0].annotate(model_name, (x_params[i], distill_acc[i]), fontsize=8, xytext=(4, -10), textcoords="offset points")
    axes[0].set_xlabel("Parameter Count")
    axes[0].set_ylabel("Test Accuracy")
    axes[0].set_ylim(0.0, 1.0)
    axes[0].set_title("Accuracy vs Parameter Count")
    axes[0].legend()

    axes[1].scatter(x_params, baseline_f1, label="baseline", color="#4C78A8", s=70)
    axes[1].scatter(x_params, distill_f1, label="distillation", color="#F58518", s=70)
    for i, model_name in enumerate(x_model):
        axes[1].annotate(model_name, (x_params[i], baseline_f1[i]), fontsize=8, xytext=(4, 4), textcoords="offset points")
        axes[1].annotate(model_name, (x_params[i], distill_f1[i]), fontsize=8, xytext=(4, -10), textcoords="offset points")
    axes[1].set_xlabel("Parameter Count")
    axes[1].set_ylabel("Test Macro-F1")
    axes[1].set_ylim(0.0, 1.0)
    axes[1].set_title("Macro-F1 vs Parameter Count")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(output_dir / "acc_by_params.png", dpi=140)
    plt.close(fig)

    print(json.dumps(rows, indent=2))
    print(f"Saved: {output_dir / 'summary.json'}")
    print(f"Saved: {output_dir / 'summary.csv'}")
    print(f"Saved: {output_dir / 'acc_by_model.png'}")
    print(f"Saved: {output_dir / 'acc_by_params.png'}")


if __name__ == "__main__":
    main()
