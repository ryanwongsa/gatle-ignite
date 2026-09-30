import torch.nn as nn


class Model(nn.Module):
    def __init__(self, in_dim=64, hidden=128, n_classes=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.BatchNorm1d(hidden),  # gives the EMA a non-floating buffer to cope with
            nn.Linear(hidden, n_classes),
        )

    def forward(self, x):
        return {"logits": self.net(x)}
