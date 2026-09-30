"""Two optimizers behind the framework's one-optimizer contract.

Not a torch optimizer: it implements only the surface the framework touches.
"""

import torch


class GanOptimizers:
    def __init__(self, gen_opt, disc_opt):
        self.gen = gen_opt
        self.disc = disc_opt

    @property
    def param_groups(self):
        # G's groups first, so train/lr_0 is the generator.
        return list(self.gen.param_groups) + list(self.disc.param_groups)

    def state_dict(self):
        return {"gen": self.gen.state_dict(), "disc": self.disc.state_dict()}

    def load_state_dict(self, state):
        self.gen.load_state_dict(state["gen"])
        self.disc.load_state_dict(state["disc"])

    def zero_grad(self, set_to_none=True):
        self.gen.zero_grad(set_to_none=set_to_none)
        self.disc.zero_grad(set_to_none=set_to_none)

    def step(self, *args, **kwargs):
        # Driven as one optimizer, this would step G on D's gradients: refuse, don't train wrong.
        raise RuntimeError(
            "GanOptimizers.step() called. A GAN's two optimizers must be stepped "
            "separately by the trainer's two-phase train_step (examples/gan/trainer/gan_trainer.py); "
            "stepping them together would apply the discriminator's update to the "
            "generator. If you see this, something is driving this object as if it were a "
            "single torch optimizer, most likely BaseTrainer.backward() or a scheduler."
        )


def get_optimizer(model, gen_lr=2e-4, disc_lr=2e-4, betas=(0.5, 0.999), **kwargs):
    """betas=(0.5, 0.999): the default 0.9 first moment makes an adversarial update oscillate."""
    # Under DDP `model.gen` raises AttributeError: reach submodules through `.module`.
    model = getattr(model, "module", model)

    # ml_collections hands tuples through as lists; torch wants a 2-tuple.
    betas = tuple(betas)
    return GanOptimizers(
        torch.optim.Adam(model.gen.parameters(), lr=gen_lr, betas=betas, **kwargs),
        torch.optim.Adam(model.disc.parameters(), lr=disc_lr, betas=betas, **kwargs),
    )
