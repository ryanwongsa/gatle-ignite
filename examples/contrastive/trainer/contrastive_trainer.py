import torch

from examples.contrastive.metrics.knn_probe import build_metric
from gatle_ignite import BaseTrainer, get_dataset, to_device


class Trainer(BaseTrainer):
    def prep_batch(self, batch, split="train", **kwargs):
        v1, v2, labels = batch
        # Only training pays for view 2: the probe sees one observation, as a downstream task would.
        model_input = {"x1": v1, "x2": v2} if split == "train" else {"x1": v1}
        # The loss never reads `labels`; the kNN metric does.
        return to_device({"model_input": model_input, "targets": {"labels": labels}})

    def build_dataloaders(self):
        """Add the kNN bank, a loader with no engine. `field=` makes a bad path name its field."""
        dls, infos = super().build_dataloaders()
        self.bank_dl, self.bank_info = get_dataset(
            self.cfg.bank_ds_name,
            self.cfg.bank_ds_params,
            self.transform,
            field="bank_ds_name",
        )
        return dls, infos

    @torch.no_grad()
    def encode_bank(self):
        """Encode the whole bank under the current weights, once per eval pass.

        model.eval() matters: in train mode BatchNorm makes features depend on batch-mates.
        """
        was_training = self.model.training
        self.model.eval()
        feats, labels = [], []
        for batch in self.bank_dl:
            prepped = self.prep_batch(batch, split="bank")
            out = self.forward(prepped["model_input"])
            feats.append(out["h1"].detach().float().cpu())
            labels.append(prepped["targets"]["labels"].detach().cpu())
        if was_training:
            self.model.train()
        return torch.cat(feats), torch.cat(labels)

    def dict_metric_from_list(self, engine_type, spec, dict_metrics, info=None):
        dict_metrics = super().dict_metric_from_list(engine_type, spec, dict_metrics, info=info)
        if engine_type == "valid":
            knn = self.cfg.get("knn_params", {})
            dict_metrics["valid/knn"] = build_metric(
                self.encode_bank,
                src_name=knn.get("src_name", "h1"),
                k=knn.get("k", 20),
                temperature=knn.get("temperature", 0.1),
            )
        return dict_metrics
