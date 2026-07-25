"""
model_def.py
Defines the CNN architecture used for disease classification.

We use a ResNet18 backbone (transfer learning) with the final FC layer
replaced for binary classification: NORMAL vs PNEUMONIA.

Swap `NUM_CLASSES` and the dataset folder structure to extend this to
multi-class disease prediction (e.g. NORMAL / BACTERIAL / VIRAL / COVID).
"""

import torch
import torch.nn as nn
from torchvision import models

NUM_CLASSES = 2
CLASS_NAMES = ["NORMAL", "PNEUMONIA"]


def build_model(num_classes: int = NUM_CLASSES, pretrained: bool = True) -> nn.Module:
    """Builds a ResNet18 with a new classification head."""
    weights = models.ResNet18_Weights.DEFAULT if pretrained else None
    model = models.resnet18(weights=weights)

    # Replace final FC layer
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model


def load_model(checkpoint_path: str, device: str = "cpu") -> nn.Module:
    """Loads a trained model checkpoint for inference."""
    model = build_model(pretrained=False)
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model
