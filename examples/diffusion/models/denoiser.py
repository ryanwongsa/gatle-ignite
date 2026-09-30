import math

import torch
from torch import nn


def timestep_embedding(t, dim):
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000.0) * torch.arange(half, dtype=torch.float32, device=t.device) / half
    )
    args = t.float().unsqueeze(1) * freqs.unsqueeze(0)
    return torch.cat([args.cos(), args.sin()], dim=1)


class Model(nn.Module):
    def __init__(self, dim=2, hidden=256, time_dim=64, n_layers=3):
        super().__init__()
        self.time_dim = time_dim
        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, hidden), nn.SiLU(), nn.Linear(hidden, hidden)
        )
        self.in_proj = nn.Linear(dim, hidden)
        self.blocks = nn.ModuleList(
            [nn.Sequential(nn.SiLU(), nn.Linear(hidden, hidden)) for _ in range(n_layers)]
        )
        self.out = nn.Sequential(nn.SiLU(), nn.Linear(hidden, dim))

    def forward(self, x_t, t):
        """x_t: (B, dim) noised points. t: (B,) integer timesteps. -> {"noise_pred": (B, dim)}"""
        h = self.in_proj(x_t) + self.time_mlp(timestep_embedding(t, self.time_dim))
        for block in self.blocks:
            h = h + block(h)
        return {"noise_pred": self.out(h)}
