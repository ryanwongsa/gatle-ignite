from contextlib import contextmanager

from examples.ema.models.model_utils.ema import EMA
from gatle_ignite import BaseTrainer, EngineSpec, to_device

# The control: the valid data, reported under "raw/", with eval_context leaving it live.
RAW_SPEC = EngineSpec.for_split("raw", ds_prefix="valid")


class Trainer(BaseTrainer):
    def prep_batch(self, batch, split="train", **kwargs):
        x, y = batch
        return to_device({"model_input": {"x": x}, "targets": {"labels": y}})

    # ---- 0. build the shadow ----

    def build_model(self):
        """Create the EMA here: extra_to_save() needs it before setup() attaches checkpoints."""
        model = super().build_model()
        self.ema = EMA(model, decay=self.cfg.get("ema_decay", 0.999))
        return model

    # ---- 1. update after every optimizer step ----

    def backward(self, loss, step=True):
        super().backward(loss, step=step)
        # No optimizer step mid-window, so nothing new to average.
        if step:
            self.ema.update(self.model)

    # ---- 2 & 3. swap the shadow in for eval, and put the live weights back ----

    def eval_specs(self):
        return super().eval_specs() + (RAW_SPEC,)

    @contextmanager
    def eval_context(self, spec):
        """The evaluator sees the shadow; the checkpoint never stores it.

        The `finally` is the safety: a leaked swap would quietly train the averaged weights.
        """
        if spec.key == RAW_SPEC.key:
            yield  # the control: live weights, same data
            return
        backup = self.ema.apply_to(self.model)
        try:
            yield
        finally:
            EMA.restore(self.model, backup)

    # ---- 4. the shadow joins the checkpoint ----

    def extra_to_save(self):
        """Covers save, resume AND `gatle-ignite eval`, which would otherwise score raw weights."""
        return {"ema": self.ema}
