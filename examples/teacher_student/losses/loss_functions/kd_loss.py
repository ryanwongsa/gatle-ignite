"""The soft term: KL(teacher_T || student_T) * T^2 (Hinton et al., 2015).

The teacher's logits arrive through `targets`, so this stays a plain config-wired sub-loss.
T^2 restores the gradient scale softmax(z/T) divides away, keeping it level with the CE term.
"""

import torch.nn as nn
import torch.nn.functional as F

from gatle_ignite import get_value


class Loss(nn.Module):
    def __init__(self, src_name="logits", tgt_name=("targets", "teacher_logits"), temperature=4.0):
        super().__init__()
        self.src_name = src_name
        self.tgt_name = tgt_name
        self.temperature = float(temperature)

    def forward(self, y_pred, target, iteration=None):
        student_logits = get_value(y_pred, self.src_name)  # (B, C)
        teacher_logits = get_value(target, self.tgt_name)  # (B, C), no grad

        t = self.temperature
        # A mismatch would broadcast, not raise, and plateau silently.
        assert student_logits.shape == teacher_logits.shape, (
            f"student {tuple(student_logits.shape)} != teacher {tuple(teacher_logits.shape)}"
        )
        student_log_p = F.log_softmax(student_logits / t, dim=-1)
        teacher_p = F.softmax(teacher_logits.detach() / t, dim=-1)
        return F.kl_div(student_log_p, teacher_p, reduction="batchmean") * (t * t)
