import torch

from examples.diffusion.models.model_utils.diffusion import Schedule
from gatle_ignite import BaseTrainer, to_device


class Trainer(BaseTrainer):
    def __init__(self, local_rank, cfg):
        super().__init__(local_rank, cfg)
        # Tiny tensors: 8 threads measured 0.30 s/step against 0.05 at 1. Set here, not in the
        # config, which is imported in the launcher's process rather than this worker's.
        threads = cfg.diffusion.get("torch_threads")
        if threads:
            torch.set_num_threads(threads)
        self.schedule = Schedule(num_steps=cfg.diffusion["num_steps"])

    def prep_batch(self, batch, split="train", **kwargs):
        (x0,) = to_device(batch)
        self.schedule.to(self.device)

        # Drawn per batch, so the same x_0 meets a new noise level every epoch.
        t = self.schedule.sample_timesteps(x0.shape[0], self.device)
        noise = torch.randn_like(x0)
        x_t = self.schedule.q_sample(x0, t, noise)

        return {
            "model_input": {"x_t": x_t, "t": t},
            # The MSE regresses onto `noise`; `x0` is the eval metric's reference.
            "targets": {"noise": noise, "x0": x0},
        }

    def eval_step(self, engine, batch, split="valid"):
        """Generate, don't predict: the batch supplies only the reference x_0 and a sample count."""
        engine.state.batch = None
        engine.state.output = None
        self.model.eval()

        x = self.prep_batch(batch, split=split)
        real = x["targets"]["x0"]

        with torch.no_grad():
            with torch.autocast(
                device_type=self.device_type, dtype=self.dtype, enabled=self.autocast_enabled
            ):

                def eps_fn(x_t, t):
                    return self.forward({"x_t": x_t, "t": t})["noise_pred"]

                # self.device: sampling starts from noise, with no input to take a device from.
                samples = self.schedule.p_sample_loop(eps_fn, tuple(real.shape), self.device)

        return {"y_pred": {"samples": samples.float()}, "target": x}
