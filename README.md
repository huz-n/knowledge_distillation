# Knowledge Distillation Project (Colab-First Starter)

This repository includes a full experimental workflow for CIFAR-10 knowledge distillation with presentation-ready outputs.

Current scope:
- Dataset: CIFAR-10
- Teacher: ResNet-50 trained on CIFAR-10, then frozen for distillation
- Student: ResNet-18 or MobileNetV3-small
- Training:
  - baseline supervised (cross-entropy)
  - embedding distillation (`mse` or `cosine`)
  - logit distillation (KL with temperature)
  - configurable weighted objective (`classification + embedding + logit`)
- Metrics:
  - accuracy
  - macro-F1
- Outputs:
  - metrics JSON/CSV
  - checkpoints
  - learning curves
  - student/model comparison plots
  - embedding PCA/t-SNE visualizations

## Quick Start (Google Colab)

1. Open `notebooks/colab_baseline_runner.ipynb` in Colab for a smoke test.
2. Open `notebooks/colab_distill_runner.ipynb` for full experiments.
3. Both notebooks mount Google Drive and cache everything under:
   - `/content/drive/MyDrive/PSI_main/knowledge_distillation` (repo)
   - `/content/drive/MyDrive/PSI_main/data` (dataset)
   - `/content/drive/MyDrive/PSI_main/cache` (torch/pip/hf caches)
   - `/content/drive/MyDrive/PSI_main/outputs` (all run artifacts)
4. Run all cells in order.
5. First run a short pass (`--max-train-batches` / `--max-val-batches`), then scale epochs/batch size.

## Local CLI Usage

```bash
python -m venv .venv
. .venv/Scripts/activate  # Windows PowerShell: .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m src.warmup_model_cache --models resnet50 resnet18 mobilenetv3_small
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
  --augment strong \
  --output-dir ./outputs/teacher_resnet50
```

Baseline:
```bash
python -m src.train_baseline \
  --student resnet18 \
  --augment strong \
  --output-dir ./outputs/baseline_core
```

Distillation (embedding + classification):
```bash
python -m src.train_distill \
  --student resnet18 \
  --teacher resnet50 \
  --teacher-checkpoint ./outputs/teacher_resnet50/teacher_best.pt \
  --augment strong \
  --embed-loss mse \
  --cls-weight 1.0 \
  --embed-weight 1.0 \
  --logit-weight 0.0 \
  --output-dir ./outputs/distill_core
```

Distillation (logit + classification):
```bash
python -m src.train_distill \
  --student resnet18 \
  --teacher resnet50 \
  --teacher-checkpoint ./outputs/teacher_resnet50/teacher_best.pt \
  --augment strong \
  --cls-weight 1.0 \
  --embed-weight 0.0 \
  --logit-weight 1.0 \
  --temperature 2.0 \
  --output-dir ./outputs/distill_logit_core
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
  --run resnet18:./outputs/baseline_resnet18/metrics.json:./outputs/distill_resnet18_embed_cls/metrics.json \
  --run mobilenetv3_small:./outputs/baseline_mobilenetv3/metrics.json:./outputs/distill_mobilenetv3_embed_cls/metrics.json \
  --output-dir ./outputs/compare_students
```

Compare distillation variants:
```bash
python -m src.compare_variants \
  --run baseline_cls:./outputs/baseline_resnet18/metrics.json \
  --run embed_plus_cls:./outputs/distill_resnet18_embed_cls/metrics.json \
  --run embed_only:./outputs/distill_resnet18_embed_only/metrics.json \
  --run logit_plus_cls:./outputs/distill_resnet18_logit_cls/metrics.json \
  --run embed_logit_cls:./outputs/distill_resnet18_embed_logit_cls/metrics.json \
  --title "ResNet-18 Distillation Ablations" \
  --output-dir ./outputs/compare_variants_resnet18
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
