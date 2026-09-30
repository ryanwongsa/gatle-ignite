"""`calls` lets checks.py prove the augmentation reaches train batches and never eval ones."""

import torch


class Transformation:
    def __init__(self, p_flip=0.5, noise=0.05, max_shift=2, brightness=0.2):
        self.p_flip = p_flip
        self.noise = noise
        self.max_shift = max_shift
        self.brightness = brightness
        self.calls = 0

    def __call__(self, x):
        self.calls += 1
        if torch.rand(1).item() < self.p_flip:
            # Horizontal flip: safe here because no class is the mirror of another.
            x = torch.flip(x, dims=(-1,))
        if self.max_shift > 0:
            sy, sx = (torch.randint(-self.max_shift, self.max_shift + 1, (2,))).tolist()
            x = torch.roll(x, shifts=(sy, sx), dims=(-2, -1))
        if self.brightness > 0:
            x = x * (1.0 + self.brightness * (2 * torch.rand(1).item() - 1))
        if self.noise > 0:
            x = x + self.noise * torch.randn_like(x)
        return x
