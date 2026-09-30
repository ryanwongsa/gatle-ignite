"""TEMPLATE: a loss term, one entry in criterion_params.dict_of_loss_params.

Each term is charted as its WEIGHTED contribution, as `train/loss_<key>_avg`.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from gatle_ignite import get_value


class Loss(nn.Module):
    """The class name must be exactly `Loss`.

    Return a BARE TENSOR: only the composite returns (total, {"loss_<key>": tensor}).
    """

    def __init__(self, src_name="logits", tgt_name=("targets", "labels"), gamma=2.0):
        super().__init__()
        # src_name/tgt_name let one loss module serve a different head without editing it.
        self.src_name = src_name
        self.tgt_name = tgt_name
        self.gamma = gamma

    def forward(self, y_pred, target, iteration=None):
        """y_pred is the model's output dict; target is prep_batch's whole return.

        `iteration` is the global step, for a term that ramps in; ignore it otherwise.
        """
        logits = get_value(y_pred, self.src_name)  # (B, C)
        labels = get_value(target, self.tgt_name)  # (B,)
        # A shape mismatch that broadcasts does not raise: it trains flat. Check both shapes.

        # Focal loss. Replace with whatever your term actually is.
        ce = F.cross_entropy(logits, labels, reduction="none")
        p_t = torch.exp(-ce)
        return ((1 - p_t) ** self.gamma * ce).mean()
