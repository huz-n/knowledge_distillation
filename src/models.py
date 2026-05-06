from __future__ import annotations

import torch
import torch.nn as nn
from torchvision.models import (
    MobileNet_V3_Small_Weights,
    ResNet18_Weights,
    ResNet50_Weights,
    mobilenet_v3_small,
    resnet18,
    resnet50,
)


class ClassifierWithEmbedding(nn.Module):
    def __init__(self, features: nn.Module, classifier: nn.Module, embedding_dim: int) -> None:
        super().__init__()
        self.features = features
        self.classifier = classifier
        self.embedding_dim = embedding_dim

    def extract_embedding(self, x: torch.Tensor) -> torch.Tensor:
        emb = self.features(x)
        if emb.ndim > 2:
            emb = torch.flatten(emb, 1)
        return emb

    def forward(self, x: torch.Tensor, return_embedding: bool = False):
        emb = self.extract_embedding(x)
        logits = self.classifier(emb)
        if return_embedding:
            return logits, emb
        return logits


def _build_resnet_with_embedding(name: str, num_classes: int, pretrained: bool) -> ClassifierWithEmbedding:
    if name == "resnet18":
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        base = resnet18(weights=weights)
    elif name == "resnet50":
        weights = ResNet50_Weights.DEFAULT if pretrained else None
        base = resnet50(weights=weights)
    else:
        raise ValueError(f"Unsupported resnet name: {name}")
    embedding_dim = base.fc.in_features
    features = nn.Sequential(*list(base.children())[:-1])
    classifier = nn.Linear(embedding_dim, num_classes)
    return ClassifierWithEmbedding(features=features, classifier=classifier, embedding_dim=embedding_dim)


def _build_mobilenetv3_small_with_embedding(
    num_classes: int, pretrained: bool
) -> ClassifierWithEmbedding:
    weights = MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
    base = mobilenet_v3_small(weights=weights)
    embedding_dim = 576
    features = nn.Sequential(
        base.features,
        base.avgpool,
    )
    classifier = nn.Linear(embedding_dim, num_classes)
    return ClassifierWithEmbedding(features=features, classifier=classifier, embedding_dim=embedding_dim)


def build_model_with_embedding(
    model_name: str, num_classes: int = 10, pretrained: bool = False
) -> ClassifierWithEmbedding:
    name = model_name.lower()
    if name in {"resnet18", "resnet50"}:
        return _build_resnet_with_embedding(name=name, num_classes=num_classes, pretrained=pretrained)
    if name in {"mobilenetv3_small", "mobilenet_v3_small"}:
        return _build_mobilenetv3_small_with_embedding(num_classes=num_classes, pretrained=pretrained)
    raise ValueError(f"Unsupported model_name: {model_name}")


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
