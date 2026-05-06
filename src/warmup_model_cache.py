from __future__ import annotations

import argparse
import os

from src.models import build_model_with_embedding


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download and cache pretrained model weights.")
    parser.add_argument(
        "--models",
        nargs="+",
        default=["resnet50", "resnet18", "mobilenetv3_small"],
        help="Model names to warm up.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(f"TORCH_HOME={os.environ.get('TORCH_HOME', '(default)')}")
    for name in args.models:
        print(f"Warming up pretrained weights for: {name}")
        _ = build_model_with_embedding(model_name=name, num_classes=10, pretrained=True)
    print("Done.")


if __name__ == "__main__":
    main()
