"""Student-teacher argmax agreement: accuracy alone cannot tell KD from a hard-label control."""

from ignite.metrics import Accuracy

from gatle_ignite import get_value


def get_metric(engine_type, info, src_name="logits", tgt_name=("targets", "teacher_logits")):
    def output_transform(output):
        student_logits = get_value(output["y_pred"], src_name)  # (B, C)
        teacher_logits = get_value(output["target"], tgt_name)  # (B, C)
        return student_logits, teacher_logits.argmax(dim=-1)  # -> Accuracy's (B,C),(B,)

    return Accuracy(output_transform=output_transform)
