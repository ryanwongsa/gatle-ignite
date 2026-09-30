"""The GAN objective as two named halves: D maximises what G minimises, so there is no total."""

import torch
import torch.nn as nn


class Loss(nn.Module):
    """Non-saturating GAN loss, as two halves.

    `label_smoothing` softens D's "real" target: the cheapest brake on D winning outright.
    """

    # One logged average per key, so train_step returns losses "loss_d" and "loss_g".
    crit_keys = ["d", "g"]

    def __init__(self, label_smoothing=0.1, **kwargs):
        super().__init__()
        self.label_smoothing = label_smoothing
        self.bce = nn.BCEWithLogitsLoss()

    def disc_loss(self, real_logits, fake_logits):
        """D wants real -> 1, fake -> 0."""
        real_target = torch.full_like(real_logits, 1.0 - self.label_smoothing)
        fake_target = torch.zeros_like(fake_logits)
        return self.bce(real_logits, real_target) + self.bce(fake_logits, fake_target)

    def gen_loss(self, fake_logits):
        """G wants D to call its fakes real.

        NON-SATURATING: the saturating form's gradient vanishes exactly when G is bad.
        """
        return self.bce(fake_logits, torch.ones_like(fake_logits))

    def forward(self, y_pred, target, iteration=None):
        raise RuntimeError(
            "GAN loss has no single total to descend: D maximises what G minimises, and "
            "the two terms need separate graphs and separate optimizer steps. "
            "examples/gan/trainer/gan_trainer.py overrides train_step and calls disc_loss()/gen_loss() "
            "directly, so this forward is never used. Reaching it means something called "
            "BaseTrainer.loss_fn(), i.e. the default supervised train_step is running and "
            "the GAN is not being trained adversarially."
        )
