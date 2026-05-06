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
from src.models import build_model_with_embedding


@dataclass
class EpochMetrics:
    epoch: int
    train_total_loss: float
    train_cls_loss: float
    train_distill_loss: float
    train_acc: float
    val_total_loss: float
    val_cls_loss: float
    val_distill_loss: float
    val_acc: float
    lr: float
    seconds: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="CIFAR-10 student training with embedding distillation."
    )
    parser.add_argument("--data-dir", type=str, default="./data")
    parser.add_argument("--output-dir", type=str, default="./outputs/distill")
    parser.add_argument("--student", type=str, default="resnet18")
    parser.add_argument("--teacher", type=str, default="resnet50")
    parser.add_argument("--teacher-checkpoint", type=str, default="")
    parser.add_argument("--teacher-pretrained", action="store_true")
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
    parser.add_argument("--distill-loss", type=str, default="mse", choices=["mse", "cosine"])
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--distill-only", action="store_true")
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_distill_loss(name: str) -> nn.Module:
    if name == "mse":
        return nn.MSELoss()
    if name == "cosine":
        return nn.CosineEmbeddingLoss()
    raise ValueError(f"Unsupported distill loss: {name}")


def distill_loss_value(
    loss_fn: nn.Module,
    student_emb: torch.Tensor,
    teacher_emb: torch.Tensor,
    loss_name: str,
) -> torch.Tensor:
    if loss_name == "mse":
        return loss_fn(student_emb, teacher_emb)
    target = torch.ones(student_emb.size(0), device=student_emb.device)
    return loss_fn(student_emb, teacher_emb, target)


def run_epoch(
    student: nn.Module,
    teacher: nn.Module,
    projector: nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    cls_criterion: nn.Module,
    distill_criterion: nn.Module,
    distill_loss_name: str,
    alpha: float,
    beta: float,
    distill_only: bool,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scaler: Optional[torch.amp.GradScaler] = None,
    max_batches: int = 0,
) -> Tuple[float, float, float, float]:
    training = optimizer is not None
    student.train(training)
    projector.train(training)
    teacher.eval()

    total_items = 0
    total_loss = 0.0
    total_cls_loss = 0.0
    total_distill_loss = 0.0
    total_correct = 0

    autocast_enabled = device.type == "cuda"
    iterator = tqdm(loader, leave=False, desc="train" if training else "eval")

    for step, (x, y) in enumerate(iterator):
        if max_batches > 0 and step >= max_batches:
            break
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        batch_size = y.size(0)

        if training and optimizer is not None:
            optimizer.zero_grad(set_to_none=True)

        with torch.no_grad():
            with torch.autocast(device_type=device.type, enabled=autocast_enabled):
                _, teacher_emb = teacher(x, return_embedding=True)

        with torch.autocast(device_type=device.type, enabled=autocast_enabled):
            student_logits, student_emb = student(x, return_embedding=True)
            proj_emb = projector(student_emb)
            cls_loss = cls_criterion(student_logits, y)
            distill_loss = distill_loss_value(
                loss_fn=distill_criterion,
                student_emb=proj_emb,
                teacher_emb=teacher_emb,
                loss_name=distill_loss_name,
            )
            total = beta * distill_loss
            if not distill_only:
                total = total + alpha * cls_loss

        if training and optimizer is not None:
            if scaler is not None and autocast_enabled:
                scaler.scale(total).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                total.backward()
                optimizer.step()

        total_items += batch_size
        total_loss += total.item() * batch_size
        total_cls_loss += cls_loss.item() * batch_size
        total_distill_loss += distill_loss.item() * batch_size
        total_correct += int((student_logits.argmax(dim=1) == y).sum().item())
        iterator.set_postfix(total=f"{total.item():.4f}")

    if total_items == 0:
        return 0.0, 0.0, 0.0, 0.0
    return (
        total_loss / total_items,
        total_cls_loss / total_items,
        total_distill_loss / total_items,
        total_correct / total_items,
    )


def evaluate_test(
    student: nn.Module,
    teacher: nn.Module,
    projector: nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    cls_criterion: nn.Module,
    distill_criterion: nn.Module,
    distill_loss_name: str,
    alpha: float,
    beta: float,
    distill_only: bool,
    max_batches: int = 0,
) -> Dict[str, float]:
    test_total, test_cls, test_distill, test_acc = run_epoch(
        student=student,
        teacher=teacher,
        projector=projector,
        loader=loader,
        device=device,
        cls_criterion=cls_criterion,
        distill_criterion=distill_criterion,
        distill_loss_name=distill_loss_name,
        alpha=alpha,
        beta=beta,
        distill_only=distill_only,
        optimizer=None,
        scaler=None,
        max_batches=max_batches,
    )
    return {
        "test_total_loss": test_total,
        "test_cls_loss": test_cls,
        "test_distill_loss": test_distill,
        "test_acc": test_acc,
    }


def plot_curves(history: List[EpochMetrics], output_path: Path) -> None:
    epochs = [h.epoch for h in history]
    train_total = [h.train_total_loss for h in history]
    val_total = [h.val_total_loss for h in history]
    train_distill = [h.train_distill_loss for h in history]
    val_distill = [h.val_distill_loss for h in history]
    val_acc = [h.val_acc for h in history]

    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    axes[0].plot(epochs, train_total, label="train_total")
    axes[0].plot(epochs, val_total, label="val_total")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Total Loss")
    axes[0].legend()

    axes[1].plot(epochs, train_distill, label="train_distill")
    axes[1].plot(epochs, val_distill, label="val_distill")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].set_title("Distillation Loss")
    axes[1].legend()

    axes[2].plot(epochs, val_acc, label="val_acc")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Accuracy")
    axes[2].set_title("Validation Accuracy")
    axes[2].legend()

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=140)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    teacher_pretrained = args.teacher_pretrained or not bool(args.teacher_checkpoint)
    teacher = build_model_with_embedding(
        model_name=args.teacher, num_classes=10, pretrained=teacher_pretrained
    ).to(device)

    if args.teacher_checkpoint:
        state = torch.load(args.teacher_checkpoint, map_location=device)
        teacher.load_state_dict(state["model"] if "model" in state else state, strict=False)
    for param in teacher.parameters():
        param.requires_grad = False
    teacher.eval()

    student = build_model_with_embedding(
        model_name=args.student, num_classes=10, pretrained=False
    ).to(device)

    projector: nn.Module
    if student.embedding_dim == teacher.embedding_dim:
        projector = nn.Identity()
    else:
        projector = nn.Linear(student.embedding_dim, teacher.embedding_dim)
    projector = projector.to(device)

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

    cls_criterion = nn.CrossEntropyLoss()
    distill_criterion = build_distill_loss(args.distill_loss)
    optimizer = AdamW(
        list(student.parameters()) + list(projector.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=max(1, args.epochs))
    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None

    history: List[EpochMetrics] = []
    best_val_acc = -1.0
    best_ckpt_path = output_dir / "student_best.pt"

    for epoch in range(1, args.epochs + 1):
        start = time.time()
        train_total, train_cls, train_distill, train_acc = run_epoch(
            student=student,
            teacher=teacher,
            projector=projector,
            loader=loaders.train,
            device=device,
            cls_criterion=cls_criterion,
            distill_criterion=distill_criterion,
            distill_loss_name=args.distill_loss,
            alpha=args.alpha,
            beta=args.beta,
            distill_only=args.distill_only,
            optimizer=optimizer,
            scaler=scaler,
            max_batches=args.max_train_batches,
        )
        val_total, val_cls, val_distill, val_acc = run_epoch(
            student=student,
            teacher=teacher,
            projector=projector,
            loader=loaders.val,
            device=device,
            cls_criterion=cls_criterion,
            distill_criterion=distill_criterion,
            distill_loss_name=args.distill_loss,
            alpha=args.alpha,
            beta=args.beta,
            distill_only=args.distill_only,
            optimizer=None,
            scaler=None,
            max_batches=args.max_val_batches,
        )
        scheduler.step()
        seconds = time.time() - start

        metrics = EpochMetrics(
            epoch=epoch,
            train_total_loss=train_total,
            train_cls_loss=train_cls,
            train_distill_loss=train_distill,
            train_acc=train_acc,
            val_total_loss=val_total,
            val_cls_loss=val_cls,
            val_distill_loss=val_distill,
            val_acc=val_acc,
            lr=float(optimizer.param_groups[0]["lr"]),
            seconds=seconds,
        )
        history.append(metrics)
        print(
            f"[Epoch {epoch}] train_total={train_total:.4f} train_cls={train_cls:.4f} "
            f"train_distill={train_distill:.4f} train_acc={train_acc:.4f} "
            f"val_total={val_total:.4f} val_acc={val_acc:.4f} time={seconds:.1f}s"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(
                {
                    "epoch": epoch,
                    "student": student.state_dict(),
                    "projector": projector.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "val_acc": val_acc,
                    "args": vars(args),
                },
                best_ckpt_path,
            )

    test_metrics = evaluate_test(
        student=student,
        teacher=teacher,
        projector=projector,
        loader=loaders.test,
        device=device,
        cls_criterion=cls_criterion,
        distill_criterion=distill_criterion,
        distill_loss_name=args.distill_loss,
        alpha=args.alpha,
        beta=args.beta,
        distill_only=args.distill_only,
        max_batches=args.max_val_batches,
    )

    plot_curves(history, output_dir / "curves.png")
    metrics_blob = {
        "args": vars(args),
        "device": str(device),
        "student_embedding_dim": student.embedding_dim,
        "teacher_embedding_dim": teacher.embedding_dim,
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
