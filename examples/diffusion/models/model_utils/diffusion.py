"""DDPM maths, outside the trainer so every caller shares one schedule."""

import math

import torch


def cosine_betas(num_steps, s=0.008, max_beta=0.999):
    """Nichol & Dhariwal's cosine schedule. Ends at alpha_bar ~ 0, so x_T really is noise.

    A linear schedule here leaves alpha_bar_T ~ 0.1, so sampling would start off-distribution.
    """
    steps = torch.arange(num_steps + 1, dtype=torch.float64)
    f = torch.cos(((steps / num_steps) + s) / (1 + s) * math.pi / 2) ** 2
    alpha_bar = f / f[0]
    betas = 1 - alpha_bar[1:] / alpha_bar[:-1]
    return betas.clamp(max=max_beta).float()


class Schedule:
    """Buffers for the forward and reverse processes, indexed by timestep."""

    def __init__(self, num_steps=200):
        self.num_steps = num_steps
        self.betas = cosine_betas(num_steps)
        self.alphas = 1.0 - self.betas
        self.alpha_bar = torch.cumprod(self.alphas, dim=0)
        self.alpha_bar_prev = torch.cat([torch.ones(1), self.alpha_bar[:-1]])

    def to(self, device):
        for name in ("betas", "alphas", "alpha_bar", "alpha_bar_prev"):
            setattr(self, name, getattr(self, name).to(device))
        return self

    def sample_timesteps(self, n, device):
        return torch.randint(0, self.num_steps, (n,), device=device)

    def q_sample(self, x0, t, noise):
        """The forward process in closed form: x_t = sqrt(abar_t) x_0 + sqrt(1 - abar_t) eps."""
        a = self.alpha_bar[t].unsqueeze(-1)
        return a.sqrt() * x0 + (1.0 - a).sqrt() * noise

    @torch.no_grad()
    def p_sample_loop(self, eps_fn, shape, device, generator=None):
        """Ancestral sampling: start at pure noise, walk t = T-1 .. 0.

        `eps_fn(x_t, t)` is a callable so the trainer can pass an autocast-wrapped forward.
        """
        x = torch.randn(shape, device=device, generator=generator)
        for step in reversed(range(self.num_steps)):
            t = torch.full((shape[0],), step, device=device, dtype=torch.long)
            eps = eps_fn(x, t)
            beta = self.betas[step]
            alpha = self.alphas[step]
            alpha_bar = self.alpha_bar[step]
            mean = (x - (beta / (1.0 - alpha_bar).sqrt()) * eps) / alpha.sqrt()
            if step > 0:
                var = beta * (1.0 - self.alpha_bar_prev[step]) / (1.0 - alpha_bar)
                noise = torch.randn(shape, device=device, generator=generator)
                x = mean + var.sqrt() * noise
            else:
                x = mean
        return x
