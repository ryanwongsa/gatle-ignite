import math

import torch
from torch.utils.data import TensorDataset

from gatle_ignite import build_dataloader

N_MODES = 8
RADIUS = 1.0
STD = 0.05


def mode_centers(n_modes=N_MODES, radius=RADIUS):
    angles = torch.arange(n_modes, dtype=torch.float32) * (2 * math.pi / n_modes)
    return torch.stack([radius * angles.cos(), radius * angles.sin()], dim=1)


def get_ds(ds_params, transform=None):
    n = ds_params.get("n", 8192)
    n_modes = ds_params.get("n_modes", N_MODES)
    radius = ds_params.get("radius", RADIUS)
    std = ds_params.get("std", STD)

    gen = torch.Generator().manual_seed(ds_params.get("seed", 0))
    centers = mode_centers(n_modes, radius)
    which = torch.randint(0, n_modes, (n,), generator=gen)
    x = centers[which] + std * torch.randn(n, 2, generator=gen)

    ds = TensorDataset(x)
    info = {
        "length": len(ds),
        # The metric learns the ring's geometry from `info`, not from the trainer or config.
        "mode_centers": centers,
        "n_modes": n_modes,
        "radius": radius,
        "std": std,
    }
    return build_dataloader(ds, ds_params), info
