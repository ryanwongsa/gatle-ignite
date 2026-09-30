import torch
from ignite.metrics import Metric
from ignite.metrics.metric import reinit__is_reduced, sync_all_reduce

from gatle_ignite import get_value


class PerClassAccuracy(Metric):
    """mean_i (correct_i / support_i). Classes with no support are excluded."""

    def __init__(
        self,
        num_classes,
        per_class=False,
        class_names=None,
        output_transform=lambda x: x,
        device="cpu",
    ):
        self.num_classes = num_classes
        self.per_class = per_class
        self.class_names = class_names or [str(i) for i in range(num_classes)]
        super().__init__(output_transform=output_transform, device=device)

    @reinit__is_reduced
    def reset(self):
        # Tensors, not python floats: sync_all_reduce can only move tensors between ranks.
        self._correct = torch.zeros(self.num_classes, device=self._device)
        self._support = torch.zeros(self.num_classes, device=self._device)
        super().reset()

    @reinit__is_reduced
    def update(self, output):
        y_pred, y = output
        y = y.to(self._device)
        pred = y_pred.argmax(dim=1).to(self._device)
        self._support += torch.bincount(y, minlength=self.num_classes).float()
        self._correct += torch.bincount(y[pred == y], minlength=self.num_classes).float()

    # Name EVERY attribute compute() reads, or the number stays rank-local under DDP.
    @sync_all_reduce("_correct", "_support")
    def compute(self):
        seen = self._support > 0
        if not bool(seen.any()):
            raise ValueError("PerClassAccuracy needs at least one example")
        # Stashed for completed(); by here sync_all_reduce has already reduced the state.
        self._acc_per_class = torch.where(
            seen, self._correct / self._support.clamp(min=1), torch.zeros_like(self._support)
        )
        return self._acc_per_class[seen].mean().item()

    def completed(self, engine, name):
        """Also publish one entry per class: ignite's dict return would need `name` as a key."""
        super().completed(engine, name)
        if self.per_class:
            for i, cls in enumerate(self.class_names):
                if self._support[i] > 0:
                    engine.state.metrics[f"{name}_{cls}"] = float(self._acc_per_class[i])


def get_metric(
    engine_type, info, src_name="logits", tgt_name=("targets", "labels"), per_class=False, **kwargs
):
    info = info or {}
    num_classes = info.get("num_classes")
    if num_classes is None:
        raise ValueError("PerClassAccuracy needs info['num_classes'] from your get_ds")

    def output_transform(output):
        return get_value(output["y_pred"], src_name), get_value(output["target"], tgt_name)

    return PerClassAccuracy(
        num_classes=num_classes,
        per_class=per_class,
        class_names=info.get("class_names"),
        output_transform=output_transform,
        **kwargs,
    )
