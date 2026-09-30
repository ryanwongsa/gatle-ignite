"""TEMPLATE: a metric for *_metrics[<name>].cls_name, reported as "valid/<name>".

The two decorators are what keep it right under DDP.
"""

import torch
from ignite.metrics import Metric
from ignite.metrics.metric import reinit__is_reduced, sync_all_reduce

from gatle_ignite import get_value


class TopKAccuracy(Metric):
    """Fraction of samples whose true label is in the model's top k. Replace with yours."""

    def __init__(self, k=2, output_transform=lambda x: x, device="cpu"):
        self.k = k
        super().__init__(output_transform=output_transform, device=device)

    # On reset() and update(): despite the name, it clears ignite's cached result, not reduces.
    @reinit__is_reduced
    def reset(self):
        # Tensors, not python floats: sync_all_reduce moves only tensors between ranks.
        self._num_correct = torch.tensor(0, device=self._device)
        self._num_examples = torch.tensor(0, device=self._device)
        super().reset()

    @reinit__is_reduced
    def update(self, output):
        y_pred, y = output  # y_pred: (B, n_classes) logits; y: (B,) labels
        topk = y_pred.topk(min(self.k, y_pred.shape[1]), dim=1).indices
        hits = (topk == y.unsqueeze(1)).any(dim=1).sum()
        self._num_correct += hits.to(self._device)
        self._num_examples += y.shape[0]

    # All-reduces these across ranks before compute(); without it each rank scores its own
    # shard. Name EVERY attribute compute() reads: a missed one stays rank-local.
    @sync_all_reduce("_num_correct", "_num_examples")
    def compute(self):
        if self._num_examples == 0:
            # Raise, don't return 0.0: a silent zero reads as a real score.
            raise ValueError("TopKAccuracy needs at least one example")
        return self._num_correct.item() / self._num_examples.item()


def get_metric(engine_type, info, src_name="logits", tgt_name=("targets", "labels"), k=2, **kwargs):
    """engine_type is "train", "valid", "test" or a declared engine's name.

    info is your get_ds's info dict: how a metric learns num_classes without the trainer.
    """
    n_classes = (info or {}).get("num_classes")
    if n_classes is not None:
        k = min(k, n_classes)

    def output_transform(output):
        # `output` is what the step returned: {"y_pred", "target"}, plus "losses" in training.
        return get_value(output["y_pred"], src_name), get_value(output["target"], tgt_name)

    return TopKAccuracy(k=k, output_transform=output_transform, **kwargs)
