"""Generator and discriminator as ONE nn.Module, so one checkpoint holds the whole GAN.

forward is the generator's, so the default eval_step samples from G with no override.
"""

import torch.nn as nn


def _mlp(in_dim, hidden, out_dim, final=None):
    layers = [
        nn.Linear(in_dim, hidden),
        nn.ReLU(),
        nn.Linear(hidden, hidden),
        nn.ReLU(),
        nn.Linear(hidden, out_dim),
    ]
    if final is not None:
        layers.append(final)
    return nn.Sequential(*layers)


class Generator(nn.Module):
    def __init__(self, z_dim=8, hidden=64, data_dim=2):
        super().__init__()
        self.net = _mlp(z_dim, hidden, data_dim)

    def forward(self, z):
        return self.net(z)


class Discriminator(nn.Module):
    def __init__(self, data_dim=2, hidden=64):
        super().__init__()
        # No sigmoid: the loss uses BCEWithLogits, which is the numerically stable form.
        self.net = _mlp(data_dim, hidden, 1)

    def forward(self, x):
        return self.net(x)


class Model(nn.Module):
    def __init__(self, z_dim=8, hidden=64, data_dim=2):
        super().__init__()
        self.z_dim = z_dim
        self.gen = Generator(z_dim=z_dim, hidden=hidden, data_dim=data_dim)
        self.disc = Discriminator(data_dim=data_dim, hidden=hidden)

    def forward(self, z):
        return {"fake": self.gen(z)}
