"""kNN probe on frozen features. NT-Xent falls even for a useless encoder, so steer by this.

The vote happens in update(), against a bank every rank encodes in full, so all-reducing
hits/total gives the single-process answer.
"""

import torch
import torch.nn.functional as F
from ignite.metrics import Metric
from ignite.metrics.metric import reinit__is_reduced, sync_all_reduce

from gatle_ignite import get_value


class KNNProbeAccuracy(Metric):
    def __init__(self, bank_fn, k=20, temperature=0.1, output_transform=lambda x: x, device="cpu"):
        # A callable, run once per eval pass, so the bank uses the CURRENT weights, not epoch 0's.
        self.bank_fn = bank_fn
        self.k = k
        self.temperature = temperature
        super().__init__(output_transform=output_transform, device=device)

    @reinit__is_reduced
    def reset(self):
        self._hits = torch.tensor(0, device=self._device)
        self._total = torch.tensor(0, device=self._device)
        self._bank_feats = None
        self._bank_labels = None
        super().reset()

    @reinit__is_reduced
    def update(self, output):
        feats, labels = output
        if self._bank_feats is None:
            bank_feats, bank_labels = self.bank_fn()
            self._bank_feats = F.normalize(bank_feats.float(), dim=-1).to(self._device)
            self._bank_labels = bank_labels.to(self._device)

        q = F.normalize(feats.float(), dim=-1).to(self._device)
        labels = labels.to(self._device)

        sim = q @ self._bank_feats.T  # (B, N_bank)
        k = min(self.k, sim.shape[1])
        top_sim, top_idx = sim.topk(k, dim=1)
        top_labels = self._bank_labels[top_idx]  # (B, k)

        n_classes = int(self._bank_labels.max().item()) + 1
        weights = (top_sim / self.temperature).exp()
        votes = torch.zeros(q.shape[0], n_classes, device=self._device)
        votes.scatter_add_(1, top_labels, weights)

        self._hits += (votes.argmax(dim=1) == labels).sum().to(self._device)
        self._total += labels.shape[0]

    @sync_all_reduce("_hits", "_total")
    def compute(self):
        if self._total == 0:
            raise ValueError("KNNProbeAccuracy saw no queries")
        return self._hits.item() / self._total.item()


def build_metric(bank_fn, src_name="h1", tgt_name=("targets", "labels"), k=20, temperature=0.1):
    """Built in Trainer.dict_metric_from_list: a config cannot hold its live bank callable."""

    def output_transform(output):
        return get_value(output["y_pred"], src_name), get_value(output["target"], tgt_name)

    return KNNProbeAccuracy(
        bank_fn, k=k, temperature=temperature, output_transform=output_transform
    )
