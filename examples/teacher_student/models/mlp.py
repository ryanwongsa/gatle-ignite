from torch import nn


class Model(nn.Module):
    def __init__(self, in_dim=64, hidden=128, depth=2, n_classes=10):
        super().__init__()
        layers, d = [], in_dim
        for _ in range(depth):
            layers += [nn.Linear(d, hidden), nn.ReLU()]
            d = hidden
        layers += [nn.Linear(d, n_classes)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return {"logits": self.net(x)}
