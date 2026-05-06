# Knowledge Distillation Project (Colab-First Starter)

This repository starts with a small, reliable baseline and now includes the first embedding-distillation core workflow.

Current scope:
- Dataset: CIFAR-10
- Teacher: ResNet-50 trained on CIFAR-10, then frozen for distillation
- Student: ResNet-18 or MobileNetV3-small
- Training:
  - baseline supervised (cross-entropy)
  - embedding distillation (`mse` or `cosine`, optional combined loss)
- Outputs: metrics JSON, checkpoint, learning curves, baseline-vs-distill comparison plot

## Quick Start (Google Colab)

1. Open `notebooks/colab_baseline_runner.ipynb` in Colab for a smoke test.
2. Open `notebooks/colab_distill_runner.ipynb` for full core runs (teacher + 2 students + plots).
3. Both notebooks mount Google Drive and cache everything under:
   - `/content/drive/MyDrive/PSI_main/knowledge_distillation` (repo)
   - `/content/drive/MyDrive/PSI_main/data` (dataset)
   - `/content/drive/MyDrive/PSI_main/cache` (torch/pip/hf caches)
   - `/content/drive/MyDrive/PSI_main/outputs` (all run artifacts)
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

## Why Teacher Checkpoint Flow Matters

Distillation quality depends on the teacher embedding space. If the teacher is not adapted to CIFAR-10,
students may match features that are less aligned with your target dataset.

Recommended flow:
1. Train teacher on CIFAR-10 and save checkpoint.
2. Freeze teacher.
3. Train students to match teacher embeddings (plus optional classification loss).

This repository supports that flow directly.

## Core Commands

Train teacher:
```bash
python -m src.train_teacher \
  --teacher-model resnet50 \
  --teacher-pretrained \
  --output-dir ./outputs/teacher_resnet50
```

Baseline:
```bash
python -m src.train_baseline --student resnet18 --output-dir ./outputs/baseline_core
```

Distillation:
```bash
python -m src.train_distill \
  --student resnet18 \
  --teacher resnet50 \
  --teacher-checkpoint ./outputs/teacher_resnet50/teacher_best.pt \
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

Compare multiple students:
```bash
python -m src.compare_students \
  --run resnet18:./outputs/baseline_resnet18/metrics.json:./outputs/distill_resnet18/metrics.json \
  --run mobilenetv3_small:./outputs/baseline_mobilenetv3/metrics.json:./outputs/distill_mobilenetv3/metrics.json \
  --output-dir ./outputs/compare_students
```

Embedding visualization:
```bash
python -m src.visualize_embeddings \
  --student-model resnet18 \
  --teacher-model resnet50 \
  --teacher-checkpoint ./outputs/teacher_resnet50/teacher_best.pt \
  --baseline-checkpoint ./outputs/baseline_resnet18/student_best.pt \
  --distill-checkpoint ./outputs/distill_resnet18/student_best.pt \
  --output-dir ./outputs/embedding_viz_resnet18
```
