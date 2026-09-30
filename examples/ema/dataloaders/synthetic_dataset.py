import torch
from torch.utils.data import TensorDataset

from gatle_ignite import build_dataloader


def get_ds(ds_params, transform=None):
    n, in_dim = ds_params.get("n", 1024), ds_params.get("in_dim", 64)
    n_classes = ds_params.get("n_classes", 10)

    weight_gen = torch.Generator().manual_seed(ds_params.get("task_seed", 1234))
    weight = torch.randn(in_dim, n_classes, generator=weight_gen)

    x_gen = torch.Generator().manual_seed(ds_params.get("seed", 0))
    x = torch.randn(n, in_dim, generator=x_gen)
    y = (x @ weight).argmax(dim=1)

    # Train-only label noise keeps late SGD bouncing, which gives the EMA something to average.
    noise = ds_params.get("label_noise", 0.0)
    if noise > 0:
        flip_gen = torch.Generator().manual_seed(ds_params.get("seed", 0) + 999)
        mask = torch.rand(n, generator=flip_gen) < noise
        y = torch.where(mask, torch.randint(0, n_classes, (n,), generator=flip_gen), y)

    ds = TensorDataset(x, y)
    return build_dataloader(ds, ds_params), {"length": len(ds)}
