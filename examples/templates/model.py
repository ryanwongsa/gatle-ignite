"""TEMPLATE: a model for cfg.model_name; model_params are splatted into Model(**model_params)."""

from torch import nn


class Model(nn.Module):
    """The class name must be exactly `Model`: that is what the framework imports."""

    def __init__(self, in_dim=64, hidden=128, n_classes=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_classes),
        )

    def forward(self, x):
        """Return a DICT, not a tensor: a config's `src_name` selects from its keys.

        prep_batch's "model_input" is splatted in, so {"x": ...} needs a parameter named `x`.
        """
        return {"logits": self.net(x)}
