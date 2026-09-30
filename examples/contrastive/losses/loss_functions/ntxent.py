import torch
import torch.nn as nn
import torch.nn.functional as F

from gatle_ignite import get_value


class Loss(nn.Module):
    def __init__(self, src_a="z1", src_b="z2", temperature=0.2):
        super().__init__()
        # Two fields, not a tuple: get_value's tuple form is a PATH into nested dicts.
        self.src_a, self.src_b = src_a, src_b
        self.temperature = temperature

    def forward(self, y_pred, target=None, iteration=None):
        z1 = get_value(y_pred, self.src_a)  # (B, D), already L2-normalised by the model
        z2 = get_value(y_pred, self.src_b)  # (B, D)
        b = z1.shape[0]
        if b < 2:
            raise ValueError(
                f"NT-Xent needs at least 2 samples per batch, got {b}: with one sample "
                "there are no negatives and the loss is identically zero."
            )

        z = torch.cat([z1, z2], dim=0)  # (2B, D)
        sim = (z @ z.T) / self.temperature  # (2B, 2B)

        # Mask the diagonal with -inf: a sample is not its own negative. Out-of-place, since
        # in-place on an autograd-tracked output risks a version-counter error.
        eye = torch.eye(2 * b, dtype=torch.bool, device=z.device)
        sim = sim.masked_fill(eye, float("-inf"))

        # THE TARGETS: i's positive is at i+B and (i+B)'s at i, indices into this batch.
        targets = torch.cat(
            [torch.arange(b, 2 * b, device=z.device), torch.arange(0, b, device=z.device)]
        )
        return F.cross_entropy(sim, targets)
