from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class MNISTDigitCNN(nn.Module):
    """
    Simple CNN: 1x28x28 -> logits(10)
    """
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B,1,28,28]
        x = self.pool(F.relu(self.conv1(x)))  # [B,32,14,14]
        x = self.pool(F.relu(self.conv2(x)))  # [B,64,7,7]
        x = x.view(x.size(0), -1)             # [B,64*7*7]
        x = F.relu(self.fc1(x))
        logits = self.fc2(x)                  # [B,10]
        return logits