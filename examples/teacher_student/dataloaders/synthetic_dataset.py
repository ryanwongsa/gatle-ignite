"""A non-linear labelling function, so a small student cannot recover it from few hard labels."""

import torch
from torch.utils.data import TensorDataset

from gatle_ignite import build_dataloader


def _make(ds_params):
    n = ds_params.get("n", 1024)
    in_dim = ds_params.get("in_dim", 64)
    n_classes = ds_params.get("n_classes", 10)

    # Task definition: identical for every split, drawn from its own generator.
    g = torch.Generator().manual_seed(ds_params.get("task_seed", 1234))
    w1 = torch.randn(in_dim, 128, generator=g)
    w2 = torch.randn(128, n_classes, generator=g)

    x_gen = torch.Generator().manual_seed(ds_params.get("seed", 0))
    x = torch.randn(n, in_dim, generator=x_gen)
    y = (torch.tanh(x @ w1 / in_dim**0.5) @ w2).argmax(dim=1)
    return TensorDataset(x, y)


def get_ds(ds_params, transform=None):
    ds = _make(ds_params)
    return build_dataloader(ds, ds_params), {"length": len(ds)}
