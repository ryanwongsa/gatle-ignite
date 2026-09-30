from gatle_ignite import BaseTrainer, to_device


class Trainer(BaseTrainer):
    def prep_batch(self, batch, split="train", **kwargs):
        model_input = {
            "src_feats": batch["src_feats"],
            "src_pad_mask": batch["src_pad_mask"],
        }
        if split == "train":
            model_input["tgt_in"] = batch["tgt_in"]
        # Otherwise omit tgt_in, so the model's default (None) selects autoregressive decoding.

        return to_device({"model_input": model_input, "targets": {"tgt_out": batch["tgt_out"]}})
