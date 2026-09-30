"""TEMPLATE: a trainer for cfg.main_runner. Usually just prep_batch: delete what you don't need."""

import torch

from gatle_ignite import BaseTrainer, EngineSpec, to_device


class Trainer(BaseTrainer):
    """The class name must be exactly `Trainer`."""

    def prep_batch(self, batch, split="train", **kwargs):
        """Map a raw batch into two dicts.

        `model_input` is splatted into the model's forward, so its keys are that signature.
        `targets` is what a config's tgt_name walks. `split` is which engine is asking.
        Keep the **kwargs, but don't read from it.
        """
        x, y = batch
        return to_device({"model_input": {"x": x}, "targets": {"labels": y}})

    # ---- Everything below is optional. Delete it unless you need it. ----

    def eval_specs(self):
        """Add an engine: its loader, metrics, cadence, logging and checkpoint score follow.

        Needs cfg.probe_ds_name, cfg.probe_metrics, cfg.every_probe.
        """
        return super().eval_specs() + (EngineSpec.for_split("probe"),)

    def backward(self, loss, step=True):
        """Custom gradient handling. This hook owns optimizer.step(), clipping and the scaler.

        Honour `step`: it is False on every batch but the last of an accumulation window.
        `loss` is already divided by the window, so do not rescale it.
        """
        loss.backward()
        for name, param in self.model.named_parameters():
            if param.grad is not None and "backbone" in name:
                param.grad *= 0.1  # e.g. a smaller effective LR for pretrained layers
        if step:
            self._clip_grads(self.model)
            self.optimizer.step()

    def forward(self, model_input):
        """Override when the call isn't a plain splat: extra args, a two-pass model."""
        return self.model(**model_input)

    def eval_step(self, engine, batch, split="valid"):
        """Override when eval is NOT a forward pass (sampling, a decode). Keep all four steps."""
        # 1. Release the last batch and output, or an epoch's worth stays alive.
        engine.state.batch = None
        engine.state.output = None
        self.model.eval()

        # 2. prep_batch is yours to call. The engine does not call it for you.
        x = self.prep_batch(batch, split=split)

        # 3. Re-enter autocast yourself, with the dtype the config resolved.
        with torch.no_grad():
            with torch.autocast(
                device_type=self.device_type, dtype=self.dtype, enabled=self.autocast_enabled
            ):
                y_pred = self.forward(x["model_input"])

        # 4. "target" is prep_batch's WHOLE return: a tgt_name is a path into it.
        return {"y_pred": y_pred, "target": x}

    def train_step(self, engine, batch, split="train"):
        """Override when the step isn't standard supervised, such as a GAN's alternating updates.

        Keep eval_step's steps. `losses` MUST carry "loss" and a `loss_<key>` per crit_keys
        entry, or the first epoch ends in a KeyError. An override drops gradient accumulation.
        """
        engine.state.batch = None
        engine.state.output = None
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)

        x = self.prep_batch(batch, split=split)
        with torch.autocast(
            device_type=self.device_type, dtype=self.dtype, enabled=self.autocast_enabled
        ):
            y_pred = self.forward(x["model_input"])
            loss, dict_losses = self.loss_fn(y_pred, x, iteration=engine.state.iteration)

        # Backward outside autocast, which is for the forward pass only.
        self.backward(loss)
        return {"y_pred": y_pred, "target": x, "losses": {"loss": loss, **dict_losses}}
