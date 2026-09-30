"""TEMPLATE: a sampler for *_ds_params.sampler_params.cls_name. Drop `shuffle` when you add one."""

import torch
from torch.utils.data import WeightedRandomSampler


def get_sampler(dataset, num_samples=None, **kwargs):
    """Return a single-process torch Sampler; under DDP the framework shards it per rank.

    Read labels OFF the dataset (a `labels` property), never by iterating it: that runs
    __getitem__, augmentation and decoding included, per sample. The fallback is slow.
    """
    if hasattr(dataset, "labels"):
        labels = torch.as_tensor([int(y) for y in dataset.labels])
    else:
        labels = torch.as_tensor([int(y) for _, y in dataset])
    counts = torch.bincount(labels).clamp(min=1)
    weights = (1.0 / counts.float())[labels]

    return WeightedRandomSampler(
        weights=weights,
        num_samples=num_samples or len(dataset),
        replacement=True,
        **kwargs,
    )
