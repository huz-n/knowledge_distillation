from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare multiple experiment variants.")
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        help="Format: label|metrics_path (preferred, Windows-safe) or legacy label:metrics_path",
    )
    parser.add_argument("--output-dir", type=str, default="./outputs/compare_variants")
    parser.add_argument("--title", type=str, default="Experiment Variant Comparison")
    return parser.parse_args()


def load_json(path: str) -> dict:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def parse_run_spec(spec: str) -> tuple[str, str]:
    if "|" in spec:
        parts = spec.split("|")
        if len(parts) != 2:
            raise ValueError(f"Invalid --run spec: {spec}")
        return parts[0], parts[1]

    parts = spec.split(":")
    if len(parts) == 2:
        return parts[0], parts[1]
    raise ValueError("Invalid --run spec. Use label|metrics_path for Windows-safe paths.")


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for spec in args.run:
        label, metrics_path = parse_run_spec(spec)
        metrics = load_json(metrics_path)
        rows.append(
            {
                "variant": label,
                "test_acc": float(metrics.get("test_acc", 0.0)),
                "test_f1": float(metrics.get("test_f1", 0.0)),
                "best_val_acc": float(metrics.get("best_val_acc", 0.0)),
            }
        )

    with (output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)
    with (output_dir / "summary.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["variant", "best_val_acc", "test_acc", "test_f1"])
        writer.writeheader()
        writer.writerows(rows)

    labels = [r["variant"] for r in rows]
    acc = [r["test_acc"] for r in rows]
    f1 = [r["test_f1"] for r in rows]
    x = np.arange(len(labels))
    width = 0.36

    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 1.4), 4.5))
    ax.bar(x - width / 2, acc, width=width, label="test_acc", color="#4C78A8")
    ax.bar(x + width / 2, f1, width=width, label="test_f1", color="#54A24B")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Score")
    ax.set_title(args.title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "variant_scores.png", dpi=140)
    plt.close(fig)

    print(json.dumps(rows, indent=2))
    print(f"Saved: {output_dir / 'summary.json'}")
    print(f"Saved: {output_dir / 'summary.csv'}")
    print(f"Saved: {output_dir / 'variant_scores.png'}")


if __name__ == "__main__":
    main()
