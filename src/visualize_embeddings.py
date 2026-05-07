from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from src.data import build_image_classification_loaders
from src.models import build_model_with_embedding


class PCATeacherTransform(nn.Module):
    def __init__(self, mean: torch.Tensor, components: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer("mean", mean)
        self.register_buffer("components", components)

    @property
    def output_dim(self) -> int:
        return int(self.components.shape[1])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) @ self.components


def build_projector_from_spec(spec: dict) -> torch.nn.Module:
    student_dim = int(spec["student_dim"])
    teacher_target_dim = int(spec["teacher_target_dim"])
    depth = int(spec.get("depth", 2))
    hidden_dim = int(spec.get("hidden_dim", 0))
    dropout = float(spec.get("dropout", 0.0))

    if depth <= 1:
        if student_dim == teacher_target_dim:
            return torch.nn.Identity()
        return torch.nn.Linear(student_dim, teacher_target_dim)

    if hidden_dim <= 0:
        hidden_dim = max(student_dim * 2, teacher_target_dim)

    layers = []
    in_dim = student_dim
    for _ in range(depth - 1):
        layers.append(torch.nn.Linear(in_dim, hidden_dim))
        layers.append(torch.nn.ReLU())
        if dropout > 0.0:
            layers.append(torch.nn.Dropout(p=dropout))
        in_dim = hidden_dim
    layers.append(torch.nn.Linear(in_dim, teacher_target_dim))
    return torch.nn.Sequential(*layers)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize teacher vs baseline vs distillation embeddings."
    )
    parser.add_argument("--data-dir", type=str, default="./data")
    parser.add_argument(
        "--dataset",
        type=str,
        default="cifar10",
        choices=["cifar10", "cifar100", "tiny_imagenet"],
    )
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
    parser.add_argument(
        "--align-mode",
        type=str,
        default="lstsq",
        choices=["none", "lstsq", "lstsq_procrustes"],
        help="How to align student embeddings to teacher space for joint visualization.",
    )
    parser.add_argument(
        "--normalize-mode",
        type=str,
        default="l2_teacher_zscore",
        choices=["none", "l2", "zscore", "l2_zscore", "teacher_zscore", "l2_teacher_zscore"],
        help=(
            "Normalization policy before PCA/t-SNE. "
            "teacher_* modes use teacher statistics as the shared reference to avoid per-model variance mismatch."
        ),
    )
    return parser.parse_args()


def load_teacher(device: torch.device, args: argparse.Namespace, num_classes: int) -> torch.nn.Module:
    teacher_pretrained = args.teacher_pretrained or not bool(args.teacher_checkpoint)
    model = build_model_with_embedding(
        model_name=args.teacher_model, num_classes=num_classes, pretrained=teacher_pretrained
    ).to(device)
    if args.teacher_checkpoint:
        state = torch.load(args.teacher_checkpoint, map_location=device)
        model.load_state_dict(state["model"] if "model" in state else state, strict=True)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model


def load_baseline_student(
    device: torch.device, args: argparse.Namespace, num_classes: int
) -> torch.nn.Module:
    model = build_model_with_embedding(model_name=args.student_model, num_classes=num_classes, pretrained=False).to(
        device
    )
    state = torch.load(args.baseline_checkpoint, map_location=device)
    model.load_state_dict(state["model"] if "model" in state else state, strict=True)
    model.eval()
    return model


def load_teacher_transform(device: torch.device, checkpoint_state: dict, teacher_emb_dim: int) -> torch.nn.Module:
    tf_state = checkpoint_state.get("teacher_transform", None)
    tf_type = checkpoint_state.get("teacher_transform_type", "Identity")
    if not tf_state:
        return torch.nn.Identity().to(device)
    if tf_type == "PCATeacherTransform":
        mean = tf_state["mean"]
        components = tf_state["components"]
        transform = PCATeacherTransform(mean=mean, components=components)
        return transform.to(device)
    if teacher_emb_dim == 0:
        raise ValueError("Unsupported teacher transform checkpoint without known teacher embedding dim.")
    return torch.nn.Identity().to(device)


def load_distill_student(
    device: torch.device,
    args: argparse.Namespace,
    checkpoint_state: dict,
    teacher_target_dim: int,
    num_classes: int,
) -> Tuple[torch.nn.Module, torch.nn.Module]:
    model = build_model_with_embedding(model_name=args.student_model, num_classes=num_classes, pretrained=False).to(
        device
    )
    state = checkpoint_state
    student_state = state["student"] if "student" in state else state.get("model", state)
    model.load_state_dict(student_state, strict=True)

    projector: torch.nn.Module
    proj_state = state.get("projector", None)
    if proj_state is None:
        if model.embedding_dim == teacher_target_dim:
            projector = torch.nn.Identity()
        else:
            projector = torch.nn.Linear(model.embedding_dim, teacher_target_dim)
    else:
        spec = state.get("projector_spec")
        if isinstance(spec, dict):
            projector = build_projector_from_spec(spec)
            projector.load_state_dict(proj_state, strict=True)
            projector = projector.to(device)
            model.eval()
            projector.eval()
            return model, projector

        keys = set(proj_state.keys())
        if len(keys) == 0:
            projector = torch.nn.Identity()
        elif "weight" in keys:
            out_dim, in_dim = proj_state["weight"].shape
            projector = torch.nn.Linear(in_dim, out_dim)
            projector.load_state_dict(proj_state, strict=True)
        elif "0.weight" in keys and "2.weight" in keys:
            h_out, in_dim = proj_state["0.weight"].shape
            out_dim, h_in = proj_state["2.weight"].shape
            if h_out != h_in:
                raise ValueError("Invalid MLP projector checkpoint dimensions.")
            projector = torch.nn.Sequential(
                torch.nn.Linear(in_dim, h_out),
                torch.nn.ReLU(),
                torch.nn.Linear(h_out, out_dim),
            )
            projector.load_state_dict(proj_state, strict=True)
        else:
            raise ValueError(
                f"Unsupported projector checkpoint format. Keys: {sorted(list(keys))[:8]}"
            )

    projector = projector.to(device)
    model.eval()
    projector.eval()
    return model, projector


def _l2_normalize_rows(x: np.ndarray) -> np.ndarray:
    denom = np.linalg.norm(x, axis=1, keepdims=True)
    denom = np.maximum(denom, 1e-12)
    return x / denom


def _zscore_features(x: np.ndarray) -> np.ndarray:
    mu = x.mean(axis=0, keepdims=True)
    sigma = x.std(axis=0, keepdims=True)
    sigma = np.maximum(sigma, 1e-6)
    return (x - mu) / sigma


def normalize_embeddings(x: np.ndarray, mode: str) -> np.ndarray:
    out = x.astype(np.float64, copy=True)
    if mode in {"l2", "l2_zscore", "l2_teacher_zscore"}:
        out = _l2_normalize_rows(out)
    if mode in {"zscore", "l2_zscore"}:
        out = _zscore_features(out)
    return out.astype(np.float32)


def _zscore_with_reference(x: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    sigma = np.maximum(sigma, 1e-6)
    return (x - mu) / sigma


def normalize_triplet(
    teacher: np.ndarray,
    baseline: np.ndarray,
    distill: np.ndarray,
    mode: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if mode in {"teacher_zscore", "l2_teacher_zscore"}:
        teacher_n = normalize_embeddings(teacher, mode="l2" if mode == "l2_teacher_zscore" else "none")
        baseline_n = normalize_embeddings(baseline, mode="l2" if mode == "l2_teacher_zscore" else "none")
        distill_n = normalize_embeddings(distill, mode="l2" if mode == "l2_teacher_zscore" else "none")
        mu = teacher_n.mean(axis=0, keepdims=True)
        sigma = teacher_n.std(axis=0, keepdims=True)
        teacher_n = _zscore_with_reference(teacher_n, mu=mu, sigma=sigma)
        baseline_n = _zscore_with_reference(baseline_n, mu=mu, sigma=sigma)
        distill_n = _zscore_with_reference(distill_n, mu=mu, sigma=sigma)
        return (
            teacher_n.astype(np.float32),
            baseline_n.astype(np.float32),
            distill_n.astype(np.float32),
        )

    return (
        normalize_embeddings(teacher, mode=mode),
        normalize_embeddings(baseline, mode=mode),
        normalize_embeddings(distill, mode=mode),
    )


def _orthogonal_procrustes_map(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    # Solves min ||A R - B||_F with R^T R = I
    m = a.T @ b
    u, _, vt = np.linalg.svd(m, full_matrices=False)
    r = u @ vt
    return r


def align_to_teacher_space(teacher: np.ndarray, student: np.ndarray, mode: str) -> np.ndarray:
    if mode == "none":
        if student.shape[1] != teacher.shape[1]:
            # Minimum fallback to make dimensions compatible in joint plots.
            w = np.linalg.lstsq(student, teacher, rcond=None)[0]
            return student @ w
        return student

    mapped = student
    if mapped.shape[1] != teacher.shape[1]:
        w = np.linalg.lstsq(mapped, teacher, rcond=None)[0]
        mapped = mapped @ w
    elif mode == "lstsq":
        w = np.linalg.lstsq(mapped, teacher, rcond=None)[0]
        mapped = mapped @ w

    if mode == "lstsq_procrustes":
        a = mapped - mapped.mean(axis=0, keepdims=True)
        b = teacher - teacher.mean(axis=0, keepdims=True)
        r = _orthogonal_procrustes_map(a, b)
        mapped = a @ r
    return mapped


@torch.no_grad()
def collect_embeddings(
    teacher: torch.nn.Module,
    teacher_transform: torch.nn.Module,
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
        t_emb = teacher_transform(t_emb)
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


def _two_dim_projection(x: np.ndarray, seed: int, perplexity: float) -> Tuple[np.ndarray, np.ndarray]:
    pca = PCA(n_components=2, random_state=seed)
    coords_pca = pca.fit_transform(x)
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        random_state=seed,
        init="pca",
        learning_rate="auto",
    )
    coords_tsne = tsne.fit_transform(x)
    return coords_pca, coords_tsne


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)

    loaders = build_image_classification_loaders(
        dataset_name=args.dataset,
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        image_size=args.image_size,
        augment="none",
        val_size=args.val_size,
        seed=args.seed,
        max_test_items=args.max_items,
    )
    num_classes = loaders.num_classes

    teacher = load_teacher(device=device, args=args, num_classes=num_classes)
    distill_state = torch.load(args.distill_checkpoint, map_location=device)
    teacher_transform = load_teacher_transform(
        device=device,
        checkpoint_state=distill_state,
        teacher_emb_dim=teacher.embedding_dim,
    )
    teacher_target_dim = (
        teacher_transform.output_dim if hasattr(teacher_transform, "output_dim") else teacher.embedding_dim
    )
    baseline_student = load_baseline_student(device=device, args=args, num_classes=num_classes)
    distill_student, distill_projector = load_distill_student(
        device=device,
        args=args,
        checkpoint_state=distill_state,
        teacher_target_dim=teacher_target_dim,
        num_classes=num_classes,
    )
    bundle = collect_embeddings(
        teacher=teacher,
        teacher_transform=teacher_transform,
        baseline_student=baseline_student,
        distill_student=distill_student,
        distill_projector=distill_projector,
        loader=loaders.test,
        device=device,
        max_items=args.max_items,
    )

    teacher_raw = bundle["teacher"]
    baseline_raw = bundle["baseline"]
    distill_raw = bundle["distill"]
    labels = bundle["labels"]

    # Fair alignment: both baseline and distilled embeddings are transformed with the same policy.
    baseline_aligned = align_to_teacher_space(
        teacher=teacher_raw, student=baseline_raw, mode=args.align_mode
    )
    distill_aligned = align_to_teacher_space(
        teacher=teacher_raw, student=distill_raw, mode=args.align_mode
    )
    teacher_aligned = teacher_raw
    if args.align_mode == "lstsq_procrustes":
        teacher_aligned = teacher_raw - teacher_raw.mean(axis=0, keepdims=True)

    teacher_arr, baseline_arr, distill_arr = normalize_triplet(
        teacher=teacher_aligned,
        baseline=baseline_aligned,
        distill=distill_aligned,
        mode=args.normalize_mode,
    )

    # Joint space plots (all models together).
    stacked = np.concatenate([teacher_arr, baseline_arr, distill_arr], axis=0)
    model_names = np.array(
        ["teacher"] * len(teacher_arr) + ["baseline"] * len(baseline_arr) + ["distill"] * len(distill_arr)
    )
    coords_pca, coords_tsne = _two_dim_projection(
        x=stacked, seed=args.seed, perplexity=args.tsne_perplexity
    )

    fig, ax = plt.subplots(figsize=(7, 5))
    _scatter_models(ax, coords_pca, model_names, "PCA (joint): Teacher vs Baseline vs Distill")
    fig.tight_layout()
    fig.savefig(output_dir / "pca_models.png", dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    _scatter_models(ax, coords_tsne, model_names, "t-SNE (joint): Teacher vs Baseline vs Distill")
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
    _scatter_classes(axes[0], teacher_pca, labels, "Teacher (PCA, joint space)")
    _scatter_classes(axes[1], baseline_pca, labels, "Baseline (PCA, joint space)")
    _scatter_classes(axes[2], distill_pca, labels, "Distill (PCA, joint space)")
    fig.tight_layout()
    fig.savefig(output_dir / "pca_by_class.png", dpi=140)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    _scatter_classes(axes[0], teacher_tsne, labels, "Teacher (t-SNE, joint space)")
    _scatter_classes(axes[1], baseline_tsne, labels, "Baseline (t-SNE, joint space)")
    _scatter_classes(axes[2], distill_tsne, labels, "Distill (t-SNE, joint space)")
    fig.tight_layout()
    fig.savefig(output_dir / "tsne_by_class.png", dpi=140)
    plt.close(fig)

    # Per-model standalone plots (not forced into one shared space).
    teacher_pca_s, teacher_tsne_s = _two_dim_projection(teacher_arr, args.seed, args.tsne_perplexity)
    baseline_pca_s, baseline_tsne_s = _two_dim_projection(baseline_arr, args.seed, args.tsne_perplexity)
    distill_pca_s, distill_tsne_s = _two_dim_projection(distill_arr, args.seed, args.tsne_perplexity)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    _scatter_classes(axes[0], teacher_pca_s, labels, "Teacher (PCA, standalone)")
    _scatter_classes(axes[1], baseline_pca_s, labels, "Baseline (PCA, standalone)")
    _scatter_classes(axes[2], distill_pca_s, labels, "Distill (PCA, standalone)")
    fig.tight_layout()
    fig.savefig(output_dir / "pca_by_class_standalone.png", dpi=140)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    _scatter_classes(axes[0], teacher_tsne_s, labels, "Teacher (t-SNE, standalone)")
    _scatter_classes(axes[1], baseline_tsne_s, labels, "Baseline (t-SNE, standalone)")
    _scatter_classes(axes[2], distill_tsne_s, labels, "Distill (t-SNE, standalone)")
    fig.tight_layout()
    fig.savefig(output_dir / "tsne_by_class_standalone.png", dpi=140)
    plt.close(fig)

    meta = {
        "num_points": int(n),
        "teacher_raw_dim": int(teacher_raw.shape[1]),
        "baseline_raw_dim": int(baseline_raw.shape[1]),
        "distill_raw_dim": int(distill_raw.shape[1]),
        "teacher_transform_type": distill_state.get("teacher_transform_type", "Identity"),
        "align_mode": args.align_mode,
        "normalize_mode": args.normalize_mode,
        "artifacts": [
            "pca_models.png",
            "tsne_models.png",
            "pca_by_class.png",
            "tsne_by_class.png",
            "pca_by_class_standalone.png",
            "tsne_by_class_standalone.png",
        ],
    }
    with (output_dir / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(json.dumps(meta, indent=2))
    print(f"Saved dir: {output_dir}")


if __name__ == "__main__":
    main()
