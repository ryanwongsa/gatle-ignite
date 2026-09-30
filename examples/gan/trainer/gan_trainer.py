"""The GAN trainer: an alternating two-optimizer train_step. The default eval_step samples G."""

import torch

from gatle_ignite import BaseTrainer, to_device


class Trainer(BaseTrainer):
    def prep_batch(self, batch, split="train", **kwargs):
        """Draw z here, not in the dataset, so its size matches a short final batch."""
        (real,) = batch
        z = torch.randn(real.shape[0], self._z_dim(), device=real.device)
        return to_device({"model_input": {"z": z}, "targets": {"real": real}})

    def train_step(self, engine, batch, split="train"):
        """Alternate: D first, so G's update is scored by the freshest critic."""
        engine.state.batch = None
        engine.state.output = None
        self.model.train()

        if self.accum_steps != 1:
            # This replaces the default train_step's accumulation: refuse a field that does nothing.
            raise NotImplementedError(
                f"accum_steps={self.accum_steps} is not supported by the GAN trainer: "
                f"this train_step replaces the framework's accumulation window. Use "
                f"accum_steps=1 and a larger bs."
            )

        gen, disc = self._nets()
        x = self.prep_batch(batch, split=split)
        real, z = x["targets"]["real"], x["model_input"]["z"]

        # ---- phase 1: the discriminator ----
        self.optimizer.disc.zero_grad(set_to_none=True)
        with self._autocast():
            # .detach() is load-bearing: without it D's backward trains G to help its critic.
            fake = gen(z).detach()
            loss_d = self.criterion.disc_loss(disc(real), disc(fake))
        loss_d.backward()
        self.optimizer.disc.step()

        # ---- phase 2: the generator ----
        self.optimizer.gen.zero_grad(set_to_none=True)
        with self._autocast():
            fake = gen(z)
            # Dirties D's .grad, harmlessly: phase 1 zeroes it, and D is not stepped here.
            loss_g = self.criterion.gen_loss(disc(fake))
        loss_g.backward()
        self.optimizer.gen.step()

        return {
            "y_pred": {"fake": fake.detach()},
            "target": x,
            # No "loss": D + G falls when EITHER wins. Watch loss_d/loss_g; trust only valid/swd.
            "losses": {
                "loss_d": loss_d.detach(),
                "loss_g": loss_g.detach(),
            },
        }

    def train_spec(self):
        """No train/loss_avg, whose D + G falls when either wins; keep loss_d_avg and loss_g_avg."""
        from dataclasses import replace

        return replace(super().train_spec(), total_loss=False)

    def _nets(self):
        """(generator, discriminator), through DDP's `.module` if present.

        The DDP path is untested: it would also need find_unused_parameters or no_sync.
        """
        model = getattr(self.model, "module", self.model)
        return model.gen, model.disc

    def _z_dim(self):
        return getattr(self.model, "module", self.model).z_dim

    def _autocast(self):
        return torch.autocast(
            device_type=self.device_type, dtype=self.dtype, enabled=self.autocast_enabled
        )
