from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from src.data import build_cifar10_loaders
from src.models import build_model_with_embedding


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize teacher vs baseline vs distillation embeddings."
    )
    parser.add_argument("--data-dir", type=str, default="./data")
    parser.add_argument("--output-dir", type=str, default="./outputs/embedding_viz")
    parser.add_argument("--student-model", type=str, default="resnet18")
    parser.add_argument("--teacher-model", type=str, default="resnet50")
    parser.add_argument("--teacher-checkpoint", type=str, default="")
    parser.add_argument("--teacher-pretrained", action="store_true")
    parser.add_argument("--baseline-checkpoint", type=str, required=True)
    parser.add_argument("--distill-checkpoint", type=str, required=True)
    parser.add_argument("--image-size", type=int, default=160)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-size", type=int, default=5000)
    parser.add_argument("--max-items", type=int, default=2000)
    parser.add_argument("--tsne-perplexity", type=float, default=30.0)
    return parser.parse_args()


def load_teacher(device: torch.device, args: argparse.Namespace) -> torch.nn.Module:
    teacher_pretrained = args.teacher_pretrained or not bool(args.teacher_checkpoint)
    model = build_model_with_embedding(
        model_name=args.teacher_model, num_classes=10, pretrained=teacher_pretrained
    ).to(device)
    if args.teacher_checkpoint:
        state = torch.load(args.teacher_checkpoint, map_location=device)
        model.load_state_dict(state["model"] if "model" in state else state, strict=False)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model


def load_baseline_student(device: torch.device, args: argparse.Namespace) -> torch.nn.Module:
    model = build_model_with_embedding(model_name=args.student_model, num_classes=10, pretrained=False).to(
        device
    )
    state = torch.load(args.baseline_checkpoint, map_location=device)
    model.load_state_dict(state["model"] if "model" in state else state, strict=False)
    model.eval()
    return model


def load_distill_student(
    device: torch.device, args: argparse.Namespace, teacher_emb_dim: int
) -> Tuple[torch.nn.Module, torch.nn.Module]:
    model = build_model_with_embedding(model_name=args.student_model, num_classes=10, pretrained=False).to(
        device
    )
    if model.embedding_dim == teacher_emb_dim:
        projector: torch.nn.Module = torch.nn.Identity()
    else:
        projector = torch.nn.Linear(model.embedding_dim, teacher_emb_dim)
    projector = projector.to(device)

    state = torch.load(args.distill_checkpoint, map_location=device)
    student_state = state["student"] if "student" in state else state.get("model", state)
    model.load_state_dict(student_state, strict=False)
    if "projector" in state:
        projector.load_state_dict(state["projector"], strict=False)
    model.eval()
    projector.eval()
    return model, projector


@torch.no_grad()
def collect_embeddings(
    teacher: torch.nn.Module,
    baseline_student: torch.nn.Module,
    distill_student: torch.nn.Module,
    distill_projector: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    max_items: int,
) -> Dict[str, np.ndarray]:
    teacher_embs = []
    baseline_embs = []
    distill_embs = []
    labels = []

    seen = 0
    for x, y in loader:
        if seen >= max_items:
            break
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        _, t_emb = teacher(x, return_embedding=True)
        _, b_emb = baseline_student(x, return_embedding=True)
        _, d_emb = distill_student(x, return_embedding=True)
        d_emb = distill_projector(d_emb)

        teacher_embs.append(t_emb.cpu())
        baseline_embs.append(b_emb.cpu())
        distill_embs.append(d_emb.cpu())
        labels.append(y.cpu())
        seen += y.size(0)

    teacher_arr = torch.cat(teacher_embs, dim=0)[:max_items].numpy()
    baseline_arr = torch.cat(baseline_embs, dim=0)[:max_items].numpy()
    distill_arr = torch.cat(distill_embs, dim=0)[:max_items].numpy()
    label_arr = torch.cat(labels, dim=0)[:max_items].numpy()

    # Align baseline to teacher space so combined plots are directly comparable.
    if baseline_arr.shape[1] != teacher_arr.shape[1]:
        x = torch.from_numpy(baseline_arr).float()
        y = torch.from_numpy(teacher_arr).float()
        solution = torch.linalg.lstsq(x, y).solution
        baseline_arr = (x @ solution).numpy()

    return {
        "teacher": teacher_arr,
        "baseline": baseline_arr,
        "distill": distill_arr,
        "labels": label_arr,
    }


def _scatter_models(ax: plt.Axes, coords: np.ndarray, names: np.ndarray, title: str) -> None:
    for name in ["teacher", "baseline", "distill"]:
        idx = names == name
        ax.scatter(coords[idx, 0], coords[idx, 1], s=8, alpha=0.5, label=name)
    ax.set_title(title)
    ax.legend()


def _scatter_classes(ax: plt.Axes, coords: np.ndarray, labels: np.ndarray, title: str) -> None:
    sc = ax.scatter(coords[:, 0], coords[:, 1], c=labels, s=8, alpha=0.6, cmap="tab10")
    ax.set_title(title)
    plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)

    teacher = load_teacher(device=device, args=args)
    baseline_student = load_baseline_student(device=device, args=args)
    distill_student, distill_projector = load_distill_student(
        device=device, args=args, teacher_emb_dim=teacher.embedding_dim
    )

    loaders = build_cifar10_loaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        image_size=args.image_size,
        val_size=args.val_size,
        seed=args.seed,
        max_test_items=args.max_items,
    )
    bundle = collect_embeddings(
        teacher=teacher,
        baseline_student=baseline_student,
        distill_student=distill_student,
        distill_projector=distill_projector,
        loader=loaders.test,
        device=device,
        max_items=args.max_items,
    )

    teacher_arr = bundle["teacher"]
    baseline_arr = bundle["baseline"]
    distill_arr = bundle["distill"]
    labels = bundle["labels"]

    stacked = np.concatenate([teacher_arr, baseline_arr, distill_arr], axis=0)
    model_names = np.array(
        ["teacher"] * len(teacher_arr) + ["baseline"] * len(baseline_arr) + ["distill"] * len(distill_arr)
    )

    pca = PCA(n_components=2, random_state=args.seed)
    coords_pca = pca.fit_transform(stacked)

    tsne = TSNE(
        n_components=2,
        perplexity=args.tsne_perplexity,
        random_state=args.seed,
        init="pca",
        learning_rate="auto",
    )
    coords_tsne = tsne.fit_transform(stacked)

    fig, ax = plt.subplots(figsize=(7, 5))
    _scatter_models(ax, coords_pca, model_names, "PCA: Teacher vs Baseline vs Distill")
    fig.tight_layout()
    fig.savefig(output_dir / "pca_models.png", dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    _scatter_models(ax, coords_tsne, model_names, "t-SNE: Teacher vs Baseline vs Distill")
    fig.tight_layout()
    fig.savefig(output_dir / "tsne_models.png", dpi=140)
    plt.close(fig)

    n = len(labels)
    teacher_pca, baseline_pca, distill_pca = coords_pca[:n], coords_pca[n : 2 * n], coords_pca[2 * n :]
    teacher_tsne, baseline_tsne, distill_tsne = (
        coords_tsne[:n],
        coords_tsne[n : 2 * n],
        coords_tsne[2 * n :],
    )

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    _scatter_classes(axes[0], teacher_pca, labels, "Teacher (PCA, by class)")
    _scatter_classes(axes[1], baseline_pca, labels, "Baseline Student (PCA, by class)")
    _scatter_classes(axes[2], distill_pca, labels, "Distill Student (PCA, by class)")
    fig.tight_layout()
    fig.savefig(output_dir / "pca_by_class.png", dpi=140)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    _scatter_classes(axes[0], teacher_tsne, labels, "Teacher (t-SNE, by class)")
    _scatter_classes(axes[1], baseline_tsne, labels, "Baseline Student (t-SNE, by class)")
    _scatter_classes(axes[2], distill_tsne, labels, "Distill Student (t-SNE, by class)")
    fig.tight_layout()
    fig.savefig(output_dir / "tsne_by_class.png", dpi=140)
    plt.close(fig)

    meta = {
        "num_points": int(n),
        "teacher_dim": int(teacher_arr.shape[1]),
        "baseline_dim_after_alignment": int(baseline_arr.shape[1]),
        "distill_dim": int(distill_arr.shape[1]),
        "artifacts": [
            "pca_models.png",
            "tsne_models.png",
            "pca_by_class.png",
            "tsne_by_class.png",
        ],
    }
    with (output_dir / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(json.dumps(meta, indent=2))
    print(f"Saved dir: {output_dir}")


if __name__ == "__main__":
    main()
