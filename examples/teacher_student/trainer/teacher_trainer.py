from gatle_ignite import BaseTrainer, to_device


class Trainer(BaseTrainer):
    def prep_batch(self, batch, split="train", **kwargs):
        x, y = batch
        return to_device({"model_input": {"x": x}, "targets": {"labels": y}})
