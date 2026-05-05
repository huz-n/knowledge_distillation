from __future__ import annotations

import torch.nn as nn
from torchvision.models import (
    MobileNet_V3_Small_Weights,
    ResNet18_Weights,
    ResNet50_Weights,
    mobilenet_v3_small,
    resnet18,
    resnet50,
)


def build_teacher_resnet50(num_classes: int = 10) -> nn.Module:
    model = resnet50(weights=ResNet50_Weights.DEFAULT)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    for param in model.parameters():
        param.requires_grad = False
    model.eval()
    return model


def build_student(student_name: str, num_classes: int = 10, pretrained: bool = False) -> nn.Module:
    name = student_name.lower()
    if name == "resnet18":
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        model = resnet18(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model
    if name in {"mobilenetv3_small", "mobilenet_v3_small"}:
        weights = MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        model = mobilenet_v3_small(weights=weights)
        model.classifier[3] = nn.Linear(model.classifier[3].in_features, num_classes)
        return model
    raise ValueError(f"Unsupported student_name: {student_name}")
