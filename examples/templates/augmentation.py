"""TEMPLATE: an augmentation for cfg.aug_name; aug_params are splatted into Transformation."""

import torch


class Transformation:
    """The class name must be exactly `Transformation`. Any callable will do.

    It is built once and handed to EVERY split: your dataset decides which ones apply it.
    """

    def __init__(self, p=0.5, scale=0.1):
        self.p = p
        self.scale = scale

    def __call__(self, x):
        if torch.rand(1).item() < self.p:
            x = x + torch.randn_like(x) * self.scale
        return x
