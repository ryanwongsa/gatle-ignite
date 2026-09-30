import torch.nn as nn

from gatle_ignite import get_value


class Loss(nn.Module):
    def __init__(self, src_name="logits", tgt_name=("targets", "labels"), **kwargs):
        super().__init__()
        self.src_name = src_name
        self.tgt_name = tgt_name
        self.loss = nn.CrossEntropyLoss(**kwargs)

    def forward(self, y_pred, target, iteration=None):
        return self.loss(get_value(y_pred, self.src_name), get_value(target, self.tgt_name))
