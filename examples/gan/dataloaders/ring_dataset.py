import torch
from torch.utils.data import TensorDataset

from gatle_ignite import build_dataloader

# Offset mean and strong correlation: an untrained generator's N(0, I)-ish blob has neither.
MEAN = torch.tensor([2.0, -1.0])
COV_L = torch.tensor([[1.2, 0.0], [1.0, 0.35]])  # cov = COV_L @ COV_L.T


def sample_real(n, seed):
    """Shared with scripts/verify.py, so its floor is drawn from the same truth."""
    gen = torch.Generator().manual_seed(seed)
    return torch.randn(n, 2, generator=gen) @ COV_L.T + MEAN


def get_ds(ds_params, transform=None):
    n = ds_params.get("n", 4096)
    x = sample_real(n, ds_params.get("seed", 0))
    ds = TensorDataset(x)
    return build_dataloader(ds, ds_params), {"length": len(ds), "data_dim": 2}
