from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt


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
            "distill_val_acc": float(distill.get("best_val_acc", 0.0)),
            "distill_test_acc": float(distill.get("test_acc", 0.0)),
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
                "distill_val_acc",
                "distill_test_acc",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    x_model = [r["model"] for r in rows]
    baseline_test = [r["baseline_test_acc"] for r in rows]
    distill_test = [r["distill_test_acc"] for r in rows]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(x_model, baseline_test, marker="o", label="baseline")
    ax.plot(x_model, distill_test, marker="o", label="distillation")
    ax.set_xlabel("Student Model")
    ax.set_ylabel("Test Accuracy")
    ax.set_ylim(0.0, 1.0)
    ax.set_title("Baseline vs Distillation (by Model)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "acc_by_model.png", dpi=140)
    plt.close(fig)

    x_params = [r["param_count"] for r in rows]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(x_params, baseline_test, marker="o", label="baseline")
    ax.plot(x_params, distill_test, marker="o", label="distillation")
    ax.set_xlabel("Parameter Count")
    ax.set_ylabel("Test Accuracy")
    ax.set_ylim(0.0, 1.0)
    ax.set_title("Baseline vs Distillation (by Parameter Count)")
    ax.legend()
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
