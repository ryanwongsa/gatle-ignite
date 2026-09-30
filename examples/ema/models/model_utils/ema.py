"""A plain object with state_dict/load_state_dict, so the shadow rides in the checkpoint."""

import torch


class EMA:
    """Exponential moving average of `model.state_dict()`, updated after every optimizer step.

    Non-floating buffers are copied, not averaged: lerping an int64 counter is a crash or a lie.
    """

    def __init__(self, model, decay=0.999, warmup=True):
        self.decay = decay
        self.warmup = warmup
        self.num_updates = 0
        # A clone, not a reference: a shadow aliasing the live tensors always equals the model.
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()}

    def _decay(self):
        if not self.warmup:
            return self.decay
        # Warmup: follow the live weights early, or the EMA is mostly the random init.
        return min(self.decay, (1 + self.num_updates) / (10 + self.num_updates))

    @torch.no_grad()
    def update(self, model):
        d = self._decay()
        for key, live in model.state_dict().items():
            shadow = self.shadow[key]
            if shadow.is_floating_point():
                shadow.mul_(d).add_(live.detach(), alpha=1.0 - d)
            else:
                shadow.copy_(live)
        self.num_updates += 1

    def apply_to(self, model):
        """Put the shadow into `model`, returning the live weights for `restore`."""
        backup = {k: v.detach().clone() for k, v in model.state_dict().items()}
        model.load_state_dict(self.shadow, strict=True)
        return backup

    @staticmethod
    def restore(model, backup):
        model.load_state_dict(backup, strict=True)

    def state_dict(self):
        return {
            "decay": self.decay,
            "warmup": self.warmup,
            "num_updates": self.num_updates,
            "shadow": self.shadow,
        }

    def load_state_dict(self, state_dict):
        self.decay = state_dict["decay"]
        self.warmup = state_dict.get("warmup", True)
        self.num_updates = state_dict["num_updates"]
        # In place, so the shadow stays on the model's device despite map_location="cpu".
        for key, value in state_dict["shadow"].items():
            self.shadow[key].copy_(value)
