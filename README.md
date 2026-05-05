# Knowledge Distillation Project (Colab-First Starter)

This repository starts with a small, reliable baseline to validate the full training pipeline before we add embedding distillation complexity.

Current scope:
- Dataset: CIFAR-10
- Teacher: pretrained ResNet-50 (frozen, loaded for compatibility checks)
- Student: ResNet-18 (trainable baseline)
- Training: standard supervised baseline (cross-entropy)
- Outputs: metrics, checkpoint, and learning-curve plot

## Quick Start (Google Colab)

1. Open `notebooks/colab_baseline_runner.ipynb` in Colab.
2. Run all cells in order.
3. First run a short sanity pass (`--max-train-batches` / `--max-val-batches`), then scale epochs/batch size.

## Local CLI Usage

```bash
python -m venv .venv
. .venv/Scripts/activate  # Windows PowerShell: .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m src.train_baseline --epochs 1 --max-train-batches 20 --max-val-batches 10
```

Artifacts are stored in `outputs/`.

## Next Step Plan

After baseline is stable, we add:
1. Student embedding head + projection module.
2. Distillation loss (MSE / cosine).
3. Baseline vs distillation experiment runner and comparison plots.
