# Knowledge Distillation Project Documentation

## 1. Project Purpose

This project investigates knowledge distillation for image classification, with emphasis on feature/embedding distillation and comparison against standard supervised baselines.

Primary goals:
- Train one larger teacher model and smaller student models.
- Compare baseline student training vs distillation variants under controlled settings.
- Measure effects on `accuracy` and `macro-F1`.
- Visualize learning dynamics and embedding spaces.

This document is intended as a full handoff for preparing a final presentation/report.

---

## 2. Repository Structure

Core code:
- `src/data.py`: dataset loading, transforms, Tiny ImageNet download/extraction.
- `src/models.py`: model builders with embedding extraction support.
- `src/metrics.py`: running accuracy and macro-F1 computation.
- `src/train_teacher.py`: teacher training.
- `src/train_baseline.py`: baseline student training.
- `src/train_distill.py`: distillation training (embedding/logit/classification objective mixing).
- `src/visualize_embeddings.py`: PCA/t-SNE visualizations.
- `src/compare_runs.py`: single-model baseline vs distill comparison.
- `src/compare_students.py`: multi-student comparison tables and plots.
- `src/compare_variants.py`: ablation comparison charts.
- `src/warmup_model_cache.py`: pre-download/pre-cache pretrained weights.

Colab notebooks:
- `notebooks/colab_baseline_runner.ipynb`: smoke-test baseline pipeline.
- `notebooks/colab_distill_runner.ipynb`: CIFAR-focused distillation experiments.
- `notebooks/colab_tiny_imagenet_runner.ipynb`: harder-dataset experiments (Tiny ImageNet).

Presentation scaffolding:
- `presentation/knowledge_distillation_presentation.md`
- `presentation/speaker_notes.md`

---

## 3. Datasets and Data Pipeline

Supported datasets:
- CIFAR-10 (`10` classes)
- CIFAR-100 (`100` classes)
- Tiny ImageNet (`200` classes)

Implemented in:
- `build_image_classification_loaders(...)` in `src/data.py`

### 3.1 Split Strategy

For CIFAR and Tiny ImageNet:
- Build train set.
- Create validation split from train with deterministic seed (`random_split`).
- Keep test set separate.

### 3.2 Tiny ImageNet Specifics

Download:
- URL: `http://cs231n.stanford.edu/tiny-imagenet-200.zip`

Extraction behavior:
- Resume-friendly file extraction with progress bar (`tqdm`).
- Completion sentinel: `.extract_complete` file written after successful extraction.

Validation/test handling:
- Avoid expensive file-copy class-restructure by using custom `TinyImageNetValDataset`.
- Parse `val_annotations.txt` and map image filename -> class index directly.

### 3.3 Preprocessing and Augmentation

Transform presets:
- `none`: resize/normalize only.
- `basic`: crop + horizontal flip + normalize.
- `strong`: crop + flip + RandAugment + RandomErasing + normalize.

Exact transform definitions from `src/data.py`:
- CIFAR mean/std: `(0.4914, 0.4822, 0.4465)` / `(0.2023, 0.1994, 0.2010)`
- Tiny ImageNet mean/std: ImageNet normalization `(0.485, 0.456, 0.406)` / `(0.229, 0.224, 0.225)`
- `basic`:
  - optional resize to `image_size`
  - `RandomCrop(image_size, padding=max(4, image_size // 8))`
  - `RandomHorizontalFlip()`
  - `ToTensor()`
  - `Normalize(mean, std)`
- `strong`:
  - optional resize to `image_size`
  - `RandomCrop(image_size, padding=max(4, image_size // 8))`
  - `RandomHorizontalFlip()`
  - `RandAugment(num_ops=2, magnitude=9)`
  - `ToTensor()`
  - `Normalize(mean, std)`
  - `RandomErasing(p=0.25, scale=(0.02, 0.2), ratio=(0.3, 3.0))`

Normalization:
- CIFAR statistics for CIFAR datasets.
- ImageNet-style statistics for Tiny ImageNet.

### 3.4 Performance/Runtime Controls

DataLoader options:
- `num_workers`
- `prefetch_factor`
- `persistent_workers`
- `pin_memory` (CUDA)

Training speed controls:
- `channels_last` mode (optional).
- TF32 and cuDNN benchmark enabled for CUDA runs.

---

## 4. Model Architecture

All models use the same abstract structure:
- `backbone -> global pooling -> embedding -> classifier`

Defined by `ClassifierWithEmbedding` in `src/models.py`:
- `extract_embedding(x)` returns feature embedding.
- `forward(x, return_embedding=True)` returns `(logits, embedding)`.

### 4.1 Teacher Candidates

Current default:
- `resnet50`

Teacher training:
- `src/train_teacher.py`

Teacher usage in distillation:
- Frozen teacher during student training (`requires_grad=False`, `eval()`).

### 4.2 Student Candidates

Implemented students:
- `resnet18`
- `mobilenetv3_small`
- `tiny_cnn` (lightweight custom CNN)

Rationale:
- Include at least one significantly weaker model (`tiny_cnn`) to amplify teacher-student capacity gap.

### 4.3 Projector (Embedding Dimension Bridge)

In distillation:
- If student embedding dim equals teacher dim: identity projector.
- Else: MLP projector:
  - `Linear(student_dim -> hidden_dim)`
  - `ReLU`
  - `Linear(hidden_dim -> teacher_dim)`

where `hidden_dim = max(student_dim * 2, teacher_dim)`.

This aligns student embedding into teacher embedding space before embedding loss.

---

## 5. Distillation Objectives

Implemented in `src/train_distill.py`.

### 5.1 Classification Loss

- Cross entropy on student logits and ground-truth labels.

### 5.2 Embedding Distillation Loss

Options:
- `mse`
- `cosine` (CosineEmbeddingLoss)

Optional normalization:
- `--embed-normalize` applies L2 normalization to both projected student embedding and teacher embedding before embedding loss.

Optional lower-dimensional teacher target:
- `--distill-dim N` compresses teacher embeddings into an `N`-dimensional bottleneck before matching.
- The bottleneck is fit once per run with PCA on teacher embeddings collected from the training split.
- This is especially useful when the raw teacher space is much larger than the student space (for example, `2048 -> 128`).

### 5.3 Logit Distillation Loss

- KL divergence between temperature-scaled student and teacher logits:
  - `KL(log_softmax(student/T), softmax(teacher/T)) * T^2`

Safety rule:
- If `logit_weight > 0`, `--teacher-checkpoint` is required.
- Prevents using non-adapted/random teacher head logits as KD targets.

### 5.4 Combined Objective

Total loss:
- `L = w_cls * L_cls + w_emb * L_emb + w_logit * L_logit`

CLI weights:
- `--cls-weight`
- `--embed-weight`
- `--logit-weight`

---

## 6. Training Scripts

### 6.1 Teacher

Script:
- `python -m src.train_teacher ...`

Key options:
- `--dataset {cifar10|cifar100|tiny_imagenet}`
- `--teacher-model resnet50`
- `--teacher-pretrained`
- `--finetune-mode {full|head_only}`
- `--augment`
- `--image-size`
- `--epochs`, `--batch-size`

Implementation defaults:
- optimizer: `AdamW`
- learning rate: `3e-4`
- weight decay: `1e-4`
- scheduler: `CosineAnnealingLR(T_max=epochs)`
- CLI defaults: `batch_size=64`, `epochs=5`, `image_size=160`, `augment=basic`, `val_size=5000`
- optional runtime optimization: CUDA autocast, TF32, cuDNN benchmark, `channels_last`

Outputs:
- `teacher_best.pt`
- `metrics.json`
- `curves.png`

### 6.2 Baseline Student

Script:
- `python -m src.train_baseline ...`

Key options:
- `--dataset ...`
- `--student {resnet18|mobilenetv3_small|tiny_cnn}`
- `--augment`
- `--image-size`

Implementation defaults:
- optimizer: `AdamW`
- learning rate: `3e-4`
- weight decay: `1e-4`
- scheduler: `CosineAnnealingLR(T_max=epochs)`
- CLI defaults: `batch_size=64`, `epochs=3`, `image_size=160`, `augment=basic`, `val_size=5000`

Outputs:
- `student_best.pt`
- `metrics.json`
- `curves.png`

### 6.3 Distillation Student

Script:
- `python -m src.train_distill ...`

Key options:
- `--dataset ...`
- `--student ...`
- `--teacher ...`
- `--teacher-checkpoint ...`
- `--embed-loss {mse|cosine}`
- `--embed-normalize`
- `--cls-weight --embed-weight --logit-weight`
- `--temperature`

Implementation defaults:
- optimizer: `AdamW` over `student + projector`
- learning rate: `3e-4`
- weight decay: `1e-4`
- scheduler: `CosineAnnealingLR(T_max=epochs)`
- CLI defaults: `batch_size=64`, `epochs=3`, `image_size=160`, `augment=basic`, `val_size=5000`
- default loss mix: `cls_weight=1.0`, `embed_weight=1.0`, `logit_weight=0.0`, `temperature=2.0`
- default embedding loss: `mse`
- teacher is used during validation/test inside the distillation script only to log distillation-related losses; deployment/inference still uses student only

### 6.4 Notebook Presets vs CLI Defaults

The Colab notebooks intentionally override the lightweight CLI defaults:

- `notebooks/colab_distill_runner.ipynb`:
  - dataset: `cifar10`
  - image size: `64`
  - batch size: `96`
  - teacher epochs: `2`
  - student epochs: `3`
  - teacher mode: `head_only`
  - augment: `basic`
- `notebooks/colab_tiny_imagenet_runner.ipynb`:
  - dataset: `tiny_imagenet`
  - image size: `64`
  - batch size: `128`
  - validation split from train: `10000`
  - teacher epochs: `6`
  - student epochs: `8`
  - teacher mode: `full`
  - augment: `basic`

This distinction matters for the presentation: the codebase defaults are for a generic, quick-start CLI experience, while the notebooks define the actual experiment presets used in Colab.

Outputs:
- `student_best.pt` (includes student + projector states)
- `metrics.json`
- `curves.png`

---

## 7. Metrics and Tracking

Primary metrics:
- `accuracy`
- `macro-F1`

Computed with:
- `RunningClassificationMetrics` in `src/metrics.py`

Saved in metrics JSON for each run:
- train/validation per-epoch values
- best validation accuracy
- test accuracy/F1
- model parameter counts
- run arguments

---

## 8. Visualization and Comparison

### 8.1 Learning Curves

From trainer outputs:
- baseline/teacher curves
- distillation curves with component losses

### 8.2 Baseline vs Distillation (single model)

Script:
- `src/compare_runs.py`

Artifacts:
- `summary.json`
- `test_acc_compare.png`

### 8.3 Multi-Student Comparison

Script:
- `src/compare_students.py`

Artifacts:
- `summary.json`
- `summary.csv`
- `acc_by_model.png` (bar charts)
- `acc_by_params.png` (scatter charts)

### 8.4 Variant/Ablation Comparison

Script:
- `src/compare_variants.py`

Artifacts:
- `summary.json`
- `summary.csv`
- `variant_scores.png`

### 8.5 Embedding Visualization

Script:
- `src/visualize_embeddings.py`

Supports:
- Joint-space and standalone PCA/t-SNE outputs.
- Alignment modes: `none`, `lstsq`, `lstsq_procrustes`.
- Normalization modes: `none`, `l2`, `zscore`, `l2_zscore`.

Projector loading:
- Correctly reconstructs identity, linear, or MLP projector from checkpoint.

---

## 9. Experiment Design Guidance

### 9.1 Why Distillation May Not Beat Baseline

Observed near-equality between baseline and distill can happen when:
- Student capacity is already sufficient for dataset complexity.
- Teacher quality is not high enough above student.
- Distillation weight mix is suboptimal.
- Dataset is too easy for chosen models.

### 9.2 Effective Levers

To increase measurable distillation effect:
- Use harder dataset (Tiny ImageNet > CIFAR-10).
- Use weaker student (`tiny_cnn`) vs stronger teacher (`resnet50`).
- Keep teacher quality high (`finetune-mode full`, enough epochs).
- Tune objective weights and temperature.

### 9.3 Suggested Ablation Matrix

For one fixed student and dataset:
- Baseline: `cls-only`
- Embed-only: `w_cls=0, w_emb=1, w_logit=0`
- Embed+cls
- Logit+cls
- Embed+logit+cls
- Repeat with/without `--embed-normalize`
- Repeat for `mse` vs `cosine`

---

## 10. Reproducibility and Stability

Implemented:
- Seeded splits and sampling.
- Consistent CLI run arguments saved in metrics.
- Strict checkpoint loading for distillation-critical models.
- Tiny ImageNet extraction completion sentinel.

Important caveat:
- cuDNN benchmark and mixed precision improve speed but can introduce slight run-to-run variability.

---

## 11. Known Non-Critical Limitations

- No fully automated experiment scheduler (runs orchestrated via notebooks/commands).
- No built-in W&B integration yet.
- Presentation generation is scaffolded but final polished slide design still manual/agent-assisted.

---

## 12. High-Level Conclusion Template (to fill with final results)

Use this template after all experiments:

1. Teacher quality:
- Best teacher validation/test metrics:

2. Student comparison:
- For each student, baseline vs distill delta (accuracy/F1):

3. Ablation insights:
- Which objective combination worked best:
- Did embedding normalization help:
- Did logit KD help:

4. Representation analysis:
- Did distilled embeddings align more with teacher than baseline:

5. Final claim:
- Distillation effectiveness under what dataset/model conditions:

---

## 13. Minimal Command Recipes

Teacher:
```bash
python -m src.train_teacher \
  --dataset tiny_imagenet \
  --teacher-model resnet50 \
  --teacher-pretrained \
  --finetune-mode full \
  --augment basic \
  --image-size 64 \
  --epochs 6 \
  --batch-size 128 \
  --channels-last \
  --output-dir ./outputs_tiny/teacher_resnet50
```

Baseline:
```bash
python -m src.train_baseline \
  --dataset tiny_imagenet \
  --student tiny_cnn \
  --augment basic \
  --image-size 64 \
  --epochs 8 \
  --batch-size 128 \
  --channels-last \
  --output-dir ./outputs_tiny/baseline_tinycnn
```

Distill:
```bash
python -m src.train_distill \
  --dataset tiny_imagenet \
  --student tiny_cnn \
  --teacher resnet50 \
  --teacher-checkpoint ./outputs_tiny/teacher_resnet50/teacher_best.pt \
  --embed-loss mse \
  --embed-normalize \
  --cls-weight 1.0 \
  --embed-weight 1.0 \
  --logit-weight 0.0 \
  --temperature 2.0 \
  --augment basic \
  --image-size 64 \
  --epochs 8 \
  --batch-size 128 \
  --channels-last \
  --output-dir ./outputs_tiny/distill_tinycnn
```
