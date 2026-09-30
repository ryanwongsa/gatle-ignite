from gatle_ignite import BaseTrainer, to_device


class Trainer(BaseTrainer):
    """The whole task-specific trainer: one prep_batch."""

    def prep_batch(self, batch, split="train", **kwargs):
        image, label = batch
        return to_device({"model_input": {"image": image}, "targets": {"labels": label}})
