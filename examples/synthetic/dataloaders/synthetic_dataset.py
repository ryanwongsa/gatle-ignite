import torch
from torch.utils.data import TensorDataset

from gatle_ignite import build_dataloader


def get_ds(ds_params, transform=None):
    """A learnable synthetic task: labels are a fixed linear function of the inputs."""
    n, in_dim = ds_params.get("n", 1024), ds_params.get("in_dim", 64)
    n_classes = ds_params.get("n_classes", 10)

    # The labelling function has its own generator, so train and valid share one task.
    weight_gen = torch.Generator().manual_seed(ds_params.get("task_seed", 1234))
    weight = torch.randn(in_dim, n_classes, generator=weight_gen)

    x_gen = torch.Generator().manual_seed(ds_params.get("seed", 0))
    x = torch.randn(n, in_dim, generator=x_gen)
    y = (x @ weight).argmax(dim=1)

    ds = TensorDataset(x, y)
    return build_dataloader(ds, ds_params), {"length": len(ds)}
