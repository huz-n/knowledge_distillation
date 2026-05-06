from __future__ import annotations

import torch


class RunningClassificationMetrics:
    def __init__(self, num_classes: int) -> None:
        self.num_classes = num_classes
        self.confusion = torch.zeros((num_classes, num_classes), dtype=torch.long)
        self.total = 0
        self.correct = 0

    def update(self, logits: torch.Tensor, targets: torch.Tensor) -> None:
        preds = logits.argmax(dim=1)
        self.total += targets.numel()
        self.correct += int((preds == targets).sum().item())
        with torch.no_grad():
            k = self.num_classes
            idx = targets.to(torch.long) * k + preds.to(torch.long)
            bincount = torch.bincount(idx, minlength=k * k).reshape(k, k).cpu()
            self.confusion += bincount

    def accuracy(self) -> float:
        if self.total == 0:
            return 0.0
        return self.correct / self.total

    def macro_f1(self) -> float:
        c = self.confusion.to(torch.float32)
        tp = torch.diag(c)
        fp = c.sum(dim=0) - tp
        fn = c.sum(dim=1) - tp
        denom = (2 * tp + fp + fn).clamp_min(1e-12)
        f1 = (2 * tp) / denom
        # Ignore classes never present in targets and predictions.
        support = c.sum(dim=1) + c.sum(dim=0)
        valid = support > 0
        if valid.any():
            return float(f1[valid].mean().item())
        return 0.0
