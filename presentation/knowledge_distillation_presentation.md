# Knowledge Distillation for Image Classification
### Project Presentation (5-7 minutes)
**Author:** Nazar Huz  
**Course/Group:** PSI  
**Date:** [Fill in date]

---

## 1. Introduction

### Why this project?
- Modern vision models are accurate but expensive.
- We want smaller student models with competitive performance.
- Knowledge distillation transfers information from a larger teacher model.

### Distillation focus in this project
- Not only final labels/logits.
- Main focus: **embedding distillation** in hidden feature space.

---

## 1. Introduction (Concept)

### Distillation idea
- **Teacher model:** larger, trained, frozen.
- **Student model:** smaller, trained from scratch.
- Student learns:
  - standard classification objective, and/or
  - similarity to teacher representations.

### Expected benefit
- Better generalization for smaller models at similar compute budget.

---

## 1. Dataset

### Dataset used
- **CIFAR-10** (10 classes, natural images).
- Train / validation / test split via project pipeline.

### Preprocessing and augmentation
- Resize to `[fill: image size]`.
- Normalization with CIFAR-10 statistics.
- Augmentation presets:
  - `basic` (crop + horizontal flip),
  - `strong` (RandAugment + RandomErasing + crop/flip).

---

## 2. Project Setup

### Teacher and students
- **Teacher:** ResNet-50 (trained on CIFAR-10, then frozen).
- **Students:**
  - ResNet-18
  - MobileNetV3-small

### Why this selection?
- Clear capacity gap (teacher > students).
- Two student families for fair architecture diversity.

---

## 2. Architecture and Embedding Space

### Shared model structure
`backbone -> global pooling -> embedding -> classifier`

### Embedding distillation path
- If student and teacher embedding dimensions differ:
  - `student embedding -> linear projection -> teacher embedding space`

### Distillation targets
- Embedding loss (`MSE` / `Cosine`).
- Optional logit distillation (`KL` with temperature).

---

## 3. Experiments

### Core comparison per student
1. **Baseline classification**
2. **Embedding distillation + classification**

### Extended ablations (ResNet-18)
- Baseline classification only.
- Embedding only.
- Logit + classification.
- Embedding + logit + classification.

---

## 3. Loss Functions and Training Conditions

### Objectives
- Classification: cross-entropy
- Embedding distillation: MSE or cosine
- Logit distillation: KL divergence with temperature

### Combined objective
`L = w_cls * L_cls + w_emb * L_emb + w_logit * L_logit`

### Controlled settings
- Same optimizer, LR schedule, batch size, epochs, preprocessing family.
- Same teacher checkpoint across all student experiments.

---

## 4. Results: Quantitative

### Main metrics
- Accuracy
- Macro-F1

### Insert artifacts
- `[INSERT IMAGE] outputs/compare_students/acc_by_model.png`
- `[INSERT IMAGE] outputs/compare_students/acc_by_params.png`
- `[INSERT TABLE] outputs/compare_students/summary.csv`

### Talking points
- Which student gains more from distillation?
- Accuracy/F1 tradeoff vs parameter count.

---

## 4. Results: Learning Dynamics and Ablations

### Curves to show
- Train/validation loss
- Validation accuracy
- Validation macro-F1
- Distillation component losses (embed/logit)

### Insert artifacts
- `[INSERT IMAGE] outputs/distill_resnet18_embed_cls/curves.png`
- `[INSERT IMAGE] outputs/compare_variants_resnet18/variant_scores.png`
- `[INSERT TABLE] outputs/compare_variants_resnet18/summary.csv`

---

## 4. Results: Embedding Visualization

### Goal
- Compare representation geometry:
  - teacher vs baseline student vs distilled student

### Insert artifacts
- `[INSERT IMAGE] outputs/embedding_viz_resnet18/pca_models.png`
- `[INSERT IMAGE] outputs/embedding_viz_resnet18/tsne_models.png`
- `[INSERT IMAGE] outputs/embedding_viz_resnet18/pca_by_class.png`

### Interpretation
- Distilled embeddings should move closer to teacher structure.

---

## 5. Conclusion

### Did distillation help?
- **[Fill after final runs]**

### Which models benefited most?
- **[Fill after final runs]**

### Final takeaway
- Embedding/logit distillation can improve compact models, but gains depend on:
  - teacher quality,
  - loss weighting,
  - augmentation regime.

---

## Next Steps (Optional)

- Hyperparameter tuning (`w_cls`, `w_emb`, `w_logit`, temperature).
- Distillation from multiple teacher layers.
- Additional datasets or stronger teacher backbone.
- Experiment tracking dashboard (e.g., Weights & Biases).
