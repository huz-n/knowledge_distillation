from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm.auto import tqdm

from src.data import build_cifar10_loaders
from src.models import build_student, build_teacher_resnet50


@dataclass
class EpochMetrics:
    epoch: int
    train_loss: float
    train_acc: float
    val_loss: float
    val_acc: float
    lr: float
    seconds: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CIFAR-10 baseline student training (Colab-first).")
    parser.add_argument("--data-dir", type=str, default="./data")
    parser.add_argument("--output-dir", type=str, default="./outputs/baseline")
    parser.add_argument("--student", type=str, default="resnet18")
    parser.add_argument("--image-size", type=int, default=160)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-size", type=int, default=5000)
    parser.add_argument("--max-train-items", type=int, default=0)
    parser.add_argument("--max-val-items", type=int, default=0)
    parser.add_argument("--max-test-items", type=int, default=0)
    parser.add_argument("--max-train-batches", type=int, default=0)
    parser.add_argument("--max-val-batches", type=int, default=0)
    parser.add_argument("--load-teacher", action="store_true")
    parser.add_argument("--teacher-checkpoint", type=str, default="")
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def accuracy(logits: torch.Tensor, target: torch.Tensor) -> float:
    pred = logits.argmax(dim=1)
    correct = (pred == target).sum().item()
    return correct / target.size(0)


def run_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    criterion: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scaler: Optional[torch.amp.GradScaler] = None,
    max_batches: int = 0,
) -> Tuple[float, float]:
    training = optimizer is not None
    model.train(training)

    total_loss = 0.0
    total_correct = 0
    total_items = 0

    autocast_enabled = device.type == "cuda"
    iterator = tqdm(loader, leave=False, desc="train" if training else "eval")

    for step, (x, y) in enumerate(iterator):
        if max_batches > 0 and step >= max_batches:
            break
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.autocast(device_type=device.type, enabled=autocast_enabled):
            logits = model(x)
            loss = criterion(logits, y)

        if training and optimizer is not None:
            if scaler is not None and autocast_enabled:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

        batch_size = y.size(0)
        total_items += batch_size
        total_loss += loss.item() * batch_size
        total_correct += int((logits.argmax(dim=1) == y).sum().item())
        iterator.set_postfix(loss=f"{loss.item():.4f}")

    if total_items == 0:
        return 0.0, 0.0
    return total_loss / total_items, total_correct / total_items


def evaluate_test(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    criterion: nn.Module,
    max_batches: int = 0,
) -> Dict[str, float]:
    test_loss, test_acc = run_epoch(
        model=model,
        loader=loader,
        device=device,
        criterion=criterion,
        optimizer=None,
        scaler=None,
        max_batches=max_batches,
    )
    return {"test_loss": test_loss, "test_acc": test_acc}


def plot_curves(history: List[EpochMetrics], output_path: Path) -> None:
    epochs = [h.epoch for h in history]
    train_loss = [h.train_loss for h in history]
    val_loss = [h.val_loss for h in history]
    val_acc = [h.val_acc for h in history]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(epochs, train_loss, label="train_loss")
    axes[0].plot(epochs, val_loss, label="val_loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Loss Curves")
    axes[0].legend()

    axes[1].plot(epochs, val_acc, label="val_acc")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Validation Accuracy")
    axes[1].legend()

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=140)
    plt.close(fig)


def maybe_load_teacher(device: torch.device, checkpoint: str = "") -> None:
    teacher = build_teacher_resnet50(num_classes=10)
    teacher.to(device)
    if checkpoint:
        state = torch.load(checkpoint, map_location=device)
        teacher.load_state_dict(state["model"] if "model" in state else state, strict=False)
    teacher.eval()


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.load_teacher:
        maybe_load_teacher(device=device, checkpoint=args.teacher_checkpoint)

    loaders = build_cifar10_loaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        image_size=args.image_size,
        val_size=args.val_size,
        seed=args.seed,
        max_train_items=args.max_train_items if args.max_train_items > 0 else None,
        max_val_items=args.max_val_items if args.max_val_items > 0 else None,
        max_test_items=args.max_test_items if args.max_test_items > 0 else None,
    )

    model = build_student(student_name=args.student, num_classes=10, pretrained=False).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=max(1, args.epochs))
    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None

    history: List[EpochMetrics] = []
    best_val_acc = -1.0
    best_ckpt_path = output_dir / "student_best.pt"

    for epoch in range(1, args.epochs + 1):
        start = time.time()

        train_loss, train_acc = run_epoch(
            model=model,
            loader=loaders.train,
            device=device,
            criterion=criterion,
            optimizer=optimizer,
            scaler=scaler,
            max_batches=args.max_train_batches,
        )
        val_loss, val_acc = run_epoch(
            model=model,
            loader=loaders.val,
            device=device,
            criterion=criterion,
            optimizer=None,
            scaler=None,
            max_batches=args.max_val_batches,
        )
        scheduler.step()
        seconds = time.time() - start

        metrics = EpochMetrics(
            epoch=epoch,
            train_loss=train_loss,
            train_acc=train_acc,
            val_loss=val_loss,
            val_acc=val_acc,
            lr=float(optimizer.param_groups[0]["lr"]),
            seconds=seconds,
        )
        history.append(metrics)
        print(
            f"[Epoch {epoch}] train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} time={seconds:.1f}s"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(
                {
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "val_acc": val_acc,
                    "args": vars(args),
                },
                best_ckpt_path,
            )

    test_metrics = evaluate_test(
        model=model,
        loader=loaders.test,
        device=device,
        criterion=criterion,
        max_batches=args.max_val_batches,
    )

    plot_curves(history, output_dir / "curves.png")

    metrics_blob = {
        "args": vars(args),
        "device": str(device),
        "history": [asdict(h) for h in history],
        "best_val_acc": best_val_acc,
        **test_metrics,
    }
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics_blob, f, indent=2)

    print(f"Saved: {best_ckpt_path}")
    print(f"Saved: {output_dir / 'metrics.json'}")
    print(f"Saved: {output_dir / 'curves.png'}")


if __name__ == "__main__":
    main()
