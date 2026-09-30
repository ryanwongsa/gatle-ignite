"""TEMPLATE: a dataset for cfg.*_ds_name.

Batch size (`bs`) lives in each split's ds_params. Never set drop_last on an eval split.
"""

import torch
from torch.utils.data import Dataset

from gatle_ignite import build_dataloader


class MyDataset(Dataset):
    def __init__(self, root=".", split="train", n=1024, seed=0, transform=None):
        self.transform = transform
        self.split = split
        # Replace with your real data. Seeded: an index must return the same sample every time.
        gen = torch.Generator().manual_seed(seed)
        self.x = torch.randn(n, 64, generator=gen)
        self.y = torch.randint(0, 10, (n,), generator=gen)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        x, y = self.x[idx], self.y[idx]
        # One transform reaches every split: apply it to train only, or you augment validation.
        if self.transform is not None and self.split == "train":
            x = self.transform(x)
        return x, y


def get_ds(ds_params, transform=None):
    """Must return (dataloader, info). Keys build_dataloader doesn't read are yours to add."""
    ds = MyDataset(
        root=ds_params.get("root", "."),
        split=ds_params.get("split", "train"),
        n=ds_params.get("n", 1024),
        seed=ds_params.get("seed", 0),
        transform=transform,
    )

    # build_dataloader, not DataLoader: it shards across ranks and splits bs by world size.
    dataloader = build_dataloader(ds, ds_params)

    # "length" is required. Other keys reach every metric as its `info` (num_classes, say).
    info = {"length": len(ds), "num_classes": 10}
    return dataloader, info
