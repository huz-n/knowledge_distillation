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
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm.auto import tqdm

from src.data import build_image_classification_loaders
from src.metrics import RunningClassificationMetrics
from src.models import build_model_with_embedding


@dataclass
class EpochMetrics:
    epoch: int
    train_total_loss: float
    train_cls_loss: float
    train_embed_loss: float
    train_logit_loss: float
    train_acc: float
    train_f1: float
    val_total_loss: float
    val_cls_loss: float
    val_embed_loss: float
    val_logit_loss: float
    val_acc: float
    val_f1: float
    lr: float
    seconds: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Student training with embedding/logit distillation."
    )
    parser.add_argument("--data-dir", type=str, default="./data")
    parser.add_argument(
        "--dataset",
        type=str,
        default="cifar10",
        choices=["cifar10", "cifar100", "tiny_imagenet"],
    )
    parser.add_argument("--output-dir", type=str, default="./outputs/distill")
    parser.add_argument("--student", type=str, default="resnet18")
    parser.add_argument("--teacher", type=str, default="resnet50")
    parser.add_argument("--teacher-checkpoint", type=str, default="")
    parser.add_argument("--teacher-pretrained", action="store_true")
    parser.add_argument("--image-size", type=int, default=160)
    parser.add_argument("--augment", type=str, default="basic", choices=["none", "basic", "strong"])
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--prefetch-factor", type=int, default=4)
    parser.add_argument("--no-persistent-workers", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-size", type=int, default=5000)
    parser.add_argument("--max-train-items", type=int, default=0)
    parser.add_argument("--max-val-items", type=int, default=0)
    parser.add_argument("--max-test-items", type=int, default=0)
    parser.add_argument("--max-train-batches", type=int, default=0)
    parser.add_argument("--max-val-batches", type=int, default=0)
    parser.add_argument("--channels-last", action="store_true")
    parser.add_argument("--embed-loss", type=str, default="mse", choices=["mse", "cosine"])
    parser.add_argument(
        "--embed-normalize",
        action="store_true",
        help="L2-normalize teacher/student embeddings before embedding distillation loss.",
    )
    parser.add_argument("--cls-weight", type=float, default=1.0)
    parser.add_argument("--embed-weight", type=float, default=1.0)
    parser.add_argument("--logit-weight", type=float, default=0.0)
    parser.add_argument("--temperature", type=float, default=2.0)
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")


def build_embed_loss(name: str) -> nn.Module:
    if name == "mse":
        return nn.MSELoss()
    if name == "cosine":
        return nn.CosineEmbeddingLoss()
    raise ValueError(f"Unsupported embed loss: {name}")


def embed_loss_value(
    loss_fn: nn.Module,
    student_emb: torch.Tensor,
    teacher_emb: torch.Tensor,
    loss_name: str,
) -> torch.Tensor:
    if loss_name == "mse":
        return loss_fn(student_emb, teacher_emb)
    target = torch.ones(student_emb.size(0), device=student_emb.device)
    return loss_fn(student_emb, teacher_emb, target)


def logit_kd_loss(student_logits: torch.Tensor, teacher_logits: torch.Tensor, temperature: float) -> torch.Tensor:
    s = F.log_softmax(student_logits / temperature, dim=1)
    t = F.softmax(teacher_logits / temperature, dim=1)
    return F.kl_div(s, t, reduction="batchmean") * (temperature * temperature)


def run_epoch(
    student: nn.Module,
    teacher: nn.Module,
    projector: nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    cls_criterion: nn.Module,
    embed_criterion: nn.Module,
    embed_loss_name: str,
    cls_weight: float,
    embed_weight: float,
    logit_weight: float,
    temperature: float,
    embed_normalize: bool,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scaler: Optional[torch.amp.GradScaler] = None,
    max_batches: int = 0,
    channels_last: bool = False,
    num_classes: int = 10,
) -> Tuple[float, float, float, float, float, float]:
    training = optimizer is not None
    student.train(training)
    projector.train(training)
    teacher.eval()

    total_items = 0
    total_loss = 0.0
    total_cls_loss = 0.0
    total_embed_loss = 0.0
    total_logit_loss = 0.0
    metrics = RunningClassificationMetrics(num_classes=num_classes)

    autocast_enabled = device.type == "cuda"
    iterator = tqdm(loader, leave=False, desc="train" if training else "eval")

    for step, (x, y) in enumerate(iterator):
        if max_batches > 0 and step >= max_batches:
            break
        x = x.to(device, non_blocking=True)
        if channels_last and x.ndim == 4:
            x = x.contiguous(memory_format=torch.channels_last)
        y = y.to(device, non_blocking=True)
        batch_size = y.size(0)

        if training and optimizer is not None:
            optimizer.zero_grad(set_to_none=True)

        with torch.no_grad():
            with torch.autocast(device_type=device.type, enabled=autocast_enabled):
                teacher_logits, teacher_emb = teacher(x, return_embedding=True)

        with torch.autocast(device_type=device.type, enabled=autocast_enabled):
            student_logits, student_emb = student(x, return_embedding=True)
            proj_emb = projector(student_emb)
            if embed_normalize:
                proj_emb = F.normalize(proj_emb, dim=1)
                teacher_emb = F.normalize(teacher_emb, dim=1)

            cls_loss = cls_criterion(student_logits, y)
            embed_loss = embed_loss_value(
                loss_fn=embed_criterion,
                student_emb=proj_emb,
                teacher_emb=teacher_emb,
                loss_name=embed_loss_name,
            )
            kd_loss = logit_kd_loss(student_logits, teacher_logits, temperature=temperature)
            total = cls_weight * cls_loss + embed_weight * embed_loss + logit_weight * kd_loss

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
        total_embed_loss += embed_loss.item() * batch_size
        total_logit_loss += kd_loss.item() * batch_size
        metrics.update(logits=student_logits, targets=y)
        iterator.set_postfix(total=f"{total.item():.4f}")

    if total_items == 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    return (
        total_loss / total_items,
        total_cls_loss / total_items,
        total_embed_loss / total_items,
        total_logit_loss / total_items,
        metrics.accuracy(),
        metrics.macro_f1(),
    )


def evaluate_test(
    student: nn.Module,
    teacher: nn.Module,
    projector: nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    cls_criterion: nn.Module,
    embed_criterion: nn.Module,
    embed_loss_name: str,
    cls_weight: float,
    embed_weight: float,
    logit_weight: float,
    temperature: float,
    embed_normalize: bool,
    max_batches: int = 0,
    channels_last: bool = False,
    num_classes: int = 10,
) -> Dict[str, float]:
    test_total, test_cls, test_embed, test_logit, test_acc, test_f1 = run_epoch(
        student=student,
        teacher=teacher,
        projector=projector,
        loader=loader,
        device=device,
        cls_criterion=cls_criterion,
        embed_criterion=embed_criterion,
        embed_loss_name=embed_loss_name,
        cls_weight=cls_weight,
        embed_weight=embed_weight,
        logit_weight=logit_weight,
        temperature=temperature,
        embed_normalize=embed_normalize,
        optimizer=None,
        scaler=None,
        max_batches=max_batches,
        channels_last=channels_last,
        num_classes=num_classes,
    )
    return {
        "test_total_loss": test_total,
        "test_cls_loss": test_cls,
        "test_embed_loss": test_embed,
        "test_logit_loss": test_logit,
        "test_acc": test_acc,
        "test_f1": test_f1,
    }


def plot_curves(history: List[EpochMetrics], output_path: Path) -> None:
    epochs = [h.epoch for h in history]
    train_total = [h.train_total_loss for h in history]
    val_total = [h.val_total_loss for h in history]
    train_embed = [h.train_embed_loss for h in history]
    val_embed = [h.val_embed_loss for h in history]
    train_logit = [h.train_logit_loss for h in history]
    val_logit = [h.val_logit_loss for h in history]
    val_acc = [h.val_acc for h in history]
    val_f1 = [h.val_f1 for h in history]

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    axes[0, 0].plot(epochs, train_total, label="train_total")
    axes[0, 0].plot(epochs, val_total, label="val_total")
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Loss")
    axes[0, 0].set_title("Total Loss")
    axes[0, 0].legend()

    axes[0, 1].plot(epochs, train_embed, label="train_embed")
    axes[0, 1].plot(epochs, val_embed, label="val_embed")
    axes[0, 1].plot(epochs, train_logit, label="train_logit")
    axes[0, 1].plot(epochs, val_logit, label="val_logit")
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylabel("Loss")
    axes[0, 1].set_title("Distillation Components")
    axes[0, 1].legend()

    axes[1, 0].plot(epochs, val_acc, label="val_acc")
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].set_ylabel("Accuracy")
    axes[1, 0].set_title("Validation Accuracy")
    axes[1, 0].legend()

    axes[1, 1].plot(epochs, val_f1, label="val_macro_f1")
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].set_ylabel("F1")
    axes[1, 1].set_title("Validation Macro-F1")
    axes[1, 1].legend()

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=140)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.cls_weight == 0 and args.embed_weight == 0 and args.logit_weight == 0:
        raise ValueError("At least one of --cls-weight / --embed-weight / --logit-weight must be > 0.")
    if args.logit_weight > 0 and not args.teacher_checkpoint:
        raise ValueError(
            "Logit distillation requires --teacher-checkpoint. "
            "Without it, teacher classifier head is not dataset-trained."
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loaders = build_image_classification_loaders(
        dataset_name=args.dataset,
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        image_size=args.image_size,
        augment=args.augment,
        prefetch_factor=args.prefetch_factor,
        persistent_workers=not args.no_persistent_workers,
        val_size=args.val_size,
        seed=args.seed,
        max_train_items=args.max_train_items if args.max_train_items > 0 else None,
        max_val_items=args.max_val_items if args.max_val_items > 0 else None,
        max_test_items=args.max_test_items if args.max_test_items > 0 else None,
    )
    num_classes = loaders.num_classes

    teacher_pretrained = args.teacher_pretrained or not bool(args.teacher_checkpoint)
    teacher = build_model_with_embedding(
        model_name=args.teacher, num_classes=num_classes, pretrained=teacher_pretrained
    ).to(device)
    if args.channels_last and device.type == "cuda":
        teacher = teacher.to(memory_format=torch.channels_last)

    if args.teacher_checkpoint:
        state = torch.load(args.teacher_checkpoint, map_location=device)
        teacher.load_state_dict(state["model"] if "model" in state else state, strict=True)
    for param in teacher.parameters():
        param.requires_grad = False
    teacher.eval()

    student = build_model_with_embedding(
        model_name=args.student, num_classes=num_classes, pretrained=False
    ).to(device)
    if args.channels_last and device.type == "cuda":
        student = student.to(memory_format=torch.channels_last)
    student_total_params = sum(p.numel() for p in student.parameters())
    student_trainable_params = sum(p.numel() for p in student.parameters() if p.requires_grad)

    if student.embedding_dim == teacher.embedding_dim:
        projector: nn.Module = nn.Identity()
    else:
        hidden_dim = max(student.embedding_dim * 2, teacher.embedding_dim)
        projector = nn.Sequential(
            nn.Linear(student.embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, teacher.embedding_dim),
        )
    projector = projector.to(device)
    projector_total_params = sum(p.numel() for p in projector.parameters())

    cls_criterion = nn.CrossEntropyLoss()
    embed_criterion = build_embed_loss(args.embed_loss)
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
        train_total, train_cls, train_embed, train_logit, train_acc, train_f1 = run_epoch(
            student=student,
            teacher=teacher,
            projector=projector,
            loader=loaders.train,
            device=device,
            cls_criterion=cls_criterion,
            embed_criterion=embed_criterion,
            embed_loss_name=args.embed_loss,
            cls_weight=args.cls_weight,
            embed_weight=args.embed_weight,
            logit_weight=args.logit_weight,
            temperature=args.temperature,
            embed_normalize=args.embed_normalize,
            optimizer=optimizer,
            scaler=scaler,
            max_batches=args.max_train_batches,
            channels_last=args.channels_last,
            num_classes=num_classes,
        )
        val_total, val_cls, val_embed, val_logit, val_acc, val_f1 = run_epoch(
            student=student,
            teacher=teacher,
            projector=projector,
            loader=loaders.val,
            device=device,
            cls_criterion=cls_criterion,
            embed_criterion=embed_criterion,
            embed_loss_name=args.embed_loss,
            cls_weight=args.cls_weight,
            embed_weight=args.embed_weight,
            logit_weight=args.logit_weight,
            temperature=args.temperature,
            embed_normalize=args.embed_normalize,
            optimizer=None,
            scaler=None,
            max_batches=args.max_val_batches,
            channels_last=args.channels_last,
            num_classes=num_classes,
        )
        scheduler.step()
        seconds = time.time() - start

        metrics = EpochMetrics(
            epoch=epoch,
            train_total_loss=train_total,
            train_cls_loss=train_cls,
            train_embed_loss=train_embed,
            train_logit_loss=train_logit,
            train_acc=train_acc,
            train_f1=train_f1,
            val_total_loss=val_total,
            val_cls_loss=val_cls,
            val_embed_loss=val_embed,
            val_logit_loss=val_logit,
            val_acc=val_acc,
            val_f1=val_f1,
            lr=float(optimizer.param_groups[0]["lr"]),
            seconds=seconds,
        )
        history.append(metrics)
        print(
            f"[Epoch {epoch}] train_total={train_total:.4f} train_cls={train_cls:.4f} "
            f"train_embed={train_embed:.4f} train_logit={train_logit:.4f} "
            f"train_acc={train_acc:.4f} train_f1={train_f1:.4f} "
            f"val_total={val_total:.4f} val_acc={val_acc:.4f} val_f1={val_f1:.4f} "
            f"time={seconds:.1f}s"
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
        embed_criterion=embed_criterion,
        embed_loss_name=args.embed_loss,
        cls_weight=args.cls_weight,
        embed_weight=args.embed_weight,
        logit_weight=args.logit_weight,
        temperature=args.temperature,
        embed_normalize=args.embed_normalize,
        max_batches=args.max_val_batches,
        channels_last=args.channels_last,
        num_classes=num_classes,
    )

    plot_curves(history, output_dir / "curves.png")
    metrics_blob = {
        "args": vars(args),
        "device": str(device),
        "teacher_model": args.teacher,
        "dataset": args.dataset,
        "teacher_checkpoint_used": bool(args.teacher_checkpoint),
        "augment": args.augment,
        "embed_normalize": args.embed_normalize,
        "student_embedding_dim": student.embedding_dim,
        "teacher_embedding_dim": teacher.embedding_dim,
        "student_total_params": student_total_params,
        "student_trainable_params": student_trainable_params,
        "projector_total_params": projector_total_params,
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
