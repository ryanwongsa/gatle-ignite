import torch.nn as nn

from gatle_ignite import get_value


class Loss(nn.Module):
    def __init__(self, src_name="preds", tgt_name=("targets", "values"), **kwargs):
        super().__init__()
        self.src_name = src_name
        self.tgt_name = tgt_name
        self.loss = nn.MSELoss(**kwargs)

    def forward(self, y_pred, target, iteration=None):
        preds = get_value(y_pred, self.src_name)
        return self.loss(preds, get_value(target, self.tgt_name).to(preds.dtype))
