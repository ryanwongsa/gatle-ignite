"""Cross-entropy on (B, U, V) logits, flattened: nn.CrossEntropyLoss wants classes second.

Pads are ignored: ~20% of targets are <pad>, and learning them looks like a healthy loss.
"""

import torch.nn as nn
import torch.nn.functional as F

from examples.translation.dataloaders.data_utils.vocab import PAD_ID
from gatle_ignite import get_value


class Loss(nn.Module):
    def __init__(
        self,
        src_name="logits",
        tgt_name=("targets", "tgt_out"),
        ignore_index=PAD_ID,
        label_smoothing=0.0,
    ):
        super().__init__()
        self.src_name = src_name
        self.tgt_name = tgt_name
        self.ignore_index = ignore_index
        self.label_smoothing = label_smoothing

    def forward(self, y_pred, target, iteration=None):
        logits = get_value(y_pred, self.src_name)  # (B, U, V)
        labels = get_value(target, self.tgt_name)  # (B, U)
        return F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            labels.reshape(-1),
            ignore_index=self.ignore_index,
            label_smoothing=self.label_smoothing,
        )
