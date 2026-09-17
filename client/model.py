"""
A small CNN for Fashion-MNIST/MNIST-style 28x28 grayscale classification.

Kept intentionally tiny: per the assignment, "the main learning challenge
should come from non-IID partitioning, client dynamics, communication, and
orchestration rather than model complexity."

torch is imported lazily so that coordinator-only deployments (and the fast
unit test suite) never need it installed.
"""
from __future__ import annotations

from typing import Dict

import numpy as np


def build_model():
    import torch.nn as nn

    class FashionCNN(nn.Module):
        def __init__(self, num_classes: int = 10):
            super().__init__()
            self.conv1 = nn.Conv2d(1, 8, kernel_size=5, padding=2)
            self.conv2 = nn.Conv2d(8, 16, kernel_size=5, padding=2)
            self.pool = nn.MaxPool2d(2, 2)
            self.fc1 = nn.Linear(16 * 7 * 7, 64)
            self.fc2 = nn.Linear(64, num_classes)
            self.relu = nn.ReLU()

        def forward(self, x):
            x = self.pool(self.relu(self.conv1(x)))
            x = self.pool(self.relu(self.conv2(x)))
            x = x.flatten(1)
            x = self.relu(self.fc1(x))
            return self.fc2(x)

    return FashionCNN()


def model_contract_shapes(model=None) -> Dict[str, list]:
    """Return {param_name: shape} for the room's model contract.

    This is computed once (e.g. by the room-creation script) from a freshly
    initialized model, so the coordinator can validate every incoming update
    against it without ever importing torch itself.
    """
    if model is None:
        model = build_model()
    return {name: list(t.shape) for name, t in model.state_dict().items()}


def torch_state_to_numpy(model) -> Dict[str, np.ndarray]:
    return {name: t.detach().cpu().numpy().copy() for name, t in model.state_dict().items()}


def numpy_state_to_torch(state: Dict[str, np.ndarray], model=None):
    import torch

    if model is None:
        model = build_model()
    sd = model.state_dict()
    for name, arr in state.items():
        sd[name] = torch.from_numpy(np.array(arr)).to(sd[name].dtype)
    model.load_state_dict(sd)
    return model
