# Speaker Notes (5-7 min)

## Slide timing plan
- Slide 1: 20-25s
- Slide 2: 30-35s
- Slide 3: 35-40s
- Slide 4: 35-40s
- Slide 5: 40-45s
- Slide 6: 40-50s
- Slide 7: 50-60s
- Slide 8: 50-60s
- Slide 9: 45-55s
- Slide 10: 40-50s
- Slide 11: 20-30s

Total: ~5.5-6.5 min

## Short script cues

1. Introduction:
- State problem: large models are costly, we want compact models.
- Distillation is teacher-to-student transfer.

2. Concept:
- Emphasize that this project focuses on embedding-space transfer, not only labels.

3. Dataset:
- Mention CIFAR-10 and controlled preprocessing.

4. Setup:
- Explain teacher/student selection and why it is meaningful.

5. Architecture:
- Show pipeline and projection layer logic.

6. Experiments:
- Mention baseline vs distillation per student, then ablations on ResNet-18.

7. Loss/config:
- Explain weighted objective and identical settings for fair comparison.

8. Quantitative results:
- Read table quickly, focus on key deltas.
- Compare by model and by parameter count.

9. Dynamics/ablations:
- Show which combination worked best and whether logit loss helped.

10. Embeddings:
- Discuss visual alignment to teacher geometry.

11. Conclusion:
- Answer three required questions:
  - Did distillation help?
  - Which models benefited most?
  - Final practical conclusions.
