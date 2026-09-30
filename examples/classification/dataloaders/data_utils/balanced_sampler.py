import torch
from torch.utils.data import WeightedRandomSampler


def get_sampler(dataset, num_samples=None, num_classes=None, replacement=True):
    """Sample each class with equal probability, by weighting 1/count.

    Reads `dataset.labels`: iterating the dataset would run the augmentation per sample.
    """
    labels = torch.as_tensor(getattr(dataset, "labels", None))
    if labels.ndim == 0:
        raise ValueError("dataset exposes no `labels`; the sampler cannot weight classes")

    counts = torch.bincount(labels, minlength=num_classes or int(labels.max()) + 1).clamp(min=1)
    weights = (1.0 / counts.float())[labels]

    return WeightedRandomSampler(
        weights=weights,
        num_samples=num_samples or len(labels),
        replacement=replacement,
    )
