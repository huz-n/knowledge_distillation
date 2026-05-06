# Knowledge Distillation Project (Colab-First Starter)

This repository starts with a small, reliable baseline and now includes the first embedding-distillation core workflow.

Current scope:
- Dataset: CIFAR-10
- Teacher: pretrained/frozen ResNet-50 (embedding source)
- Student: ResNet-18 or MobileNetV3-small
- Training:
  - baseline supervised (cross-entropy)
  - embedding distillation (`mse` or `cosine`, optional combined loss)
- Outputs: metrics JSON, checkpoint, learning curves, baseline-vs-distill comparison plot

## Quick Start (Google Colab)

1. Open `notebooks/colab_baseline_runner.ipynb` in Colab for a smoke test.
2. Open `notebooks/colab_distill_runner.ipynb` for core baseline-vs-distill runs.
2. Run all cells in order.
3. First run a short sanity pass (`--max-train-batches` / `--max-val-batches`), then scale epochs/batch size.

## Local CLI Usage

```bash
python -m venv .venv
. .venv/Scripts/activate  # Windows PowerShell: .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m src.train_baseline --epochs 1 --max-train-batches 20 --max-val-batches 10
python -m src.train_distill --teacher-pretrained --epochs 1 --max-train-batches 20 --max-val-batches 10
```

Artifacts are stored in `outputs/`.

## Core Commands

Baseline:
```bash
python -m src.train_baseline --student resnet18 --output-dir ./outputs/baseline_core
```

Distillation:
```bash
python -m src.train_distill \
  --student resnet18 \
  --teacher resnet50 \
  --teacher-pretrained \
  --distill-loss mse \
  --alpha 1.0 \
  --beta 1.0 \
  --output-dir ./outputs/distill_core
```

Compare:
```bash
python -m src.compare_runs \
  --baseline-metrics ./outputs/baseline_core/metrics.json \
  --distill-metrics ./outputs/distill_core/metrics.json \
  --label resnet18 \
  --output-dir ./outputs/compare_core
```
