"""Two views share a sample's content and differ in style, mixed by one fixed matrix.

The mixing stops an encoder dropping style by ignoring inputs: it has to learn the content.
"""

import torch
from torch.utils.data import DataLoader, Dataset

from gatle_ignite import build_dataloader

# Its own generator, fixed across splits, so train, bank and query share one task.
TASK_SEED = 1234


def _task(d_content, d_style, obs_dim, n_classes, proto_scale):
    g = torch.Generator().manual_seed(TASK_SEED)
    prototypes = torch.randn(n_classes, d_content, generator=g) * proto_scale
    mixing = torch.randn(obs_dim, d_content + d_style, generator=g) / (d_content + d_style) ** 0.5
    return prototypes, mixing


class Mixer:
    """collate_fn: turns (content, style1, style2, noise1, noise2, label) into two views.

    Mixing per batch, not per item, is ~30x faster. A collate must be a pure function of its
    items: every random draw belongs in the dataset. A class, not a closure, so it pickles.
    """

    def __init__(self, mixing):
        self.mixing = mixing

    def _mix(self, content, style, noise):
        return torch.cat([content, style], dim=-1) @ self.mixing.T + noise

    def __call__(self, items):
        content, s1, s2, n1, n2, labels = (torch.stack(t) for t in zip(*items))
        return self._mix(content, s1, n1), self._mix(content, s2, n2), labels


class TwoViewDataset(Dataset):
    """`stochastic_views` (train only) redraws both styles per __getitem__: the augmentation.

    Eval splits precompute theirs, so a moving metric means a moving model.
    """

    def __init__(
        self,
        n=4096,
        n_classes=10,
        d_content=16,
        d_style=16,
        obs_dim=64,
        proto_scale=1.0,
        content_sigma=0.5,
        style_scale=4.0,
        obs_noise=0.05,
        seed=0,
        stochastic_views=False,
    ):
        self.d_style = d_style
        self.obs_dim = obs_dim
        self.style_scale = style_scale
        self.obs_noise = obs_noise
        self.stochastic_views = stochastic_views

        self.prototypes, self.mixing = _task(d_content, d_style, obs_dim, n_classes, proto_scale)
        self.mixer = Mixer(self.mixing)

        g = torch.Generator().manual_seed(seed)
        self.labels = torch.randint(0, n_classes, (n,), generator=g)
        self.content = self.prototypes[self.labels] + content_sigma * torch.randn(
            n, d_content, generator=g
        )

        if not stochastic_views:
            self.s1, self.s2 = self._styles(n, g), self._styles(n, g)
            self.n1, self.n2 = self._obs_noise(n, g), self._obs_noise(n, g)

    def _styles(self, n, g=None):
        return self.style_scale * torch.randn(n, self.d_style, generator=g)

    def _obs_noise(self, n, g=None):
        return self.obs_noise * torch.randn(n, self.obs_dim, generator=g)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        if self.stochastic_views:
            return (
                self.content[idx],
                self._styles(1)[0],
                self._styles(1)[0],
                self._obs_noise(1)[0],
                self._obs_noise(1)[0],
                self.labels[idx],
            )
        return (
            self.content[idx],
            self.s1[idx],
            self.s2[idx],
            self.n1[idx],
            self.n2[idx],
            self.labels[idx],
        )


def get_ds(ds_params, transform=None):
    """`shard=False` gives every rank the whole bank, so the kNN all-reduce is exact under DDP."""
    ds = TwoViewDataset(
        n=ds_params.get("n", 4096),
        n_classes=ds_params.get("n_classes", 10),
        d_content=ds_params.get("d_content", 16),
        d_style=ds_params.get("d_style", 16),
        obs_dim=ds_params.get("obs_dim", 64),
        proto_scale=ds_params.get("proto_scale", 1.0),
        content_sigma=ds_params.get("content_sigma", 0.5),
        style_scale=ds_params.get("style_scale", 4.0),
        seed=ds_params.get("seed", 0),
        stochastic_views=ds_params.get("stochastic_views", False),
    )

    if ds_params.get("shard", True):
        dl = build_dataloader(ds, {**ds_params, "collate_fn": ds.mixer})
    else:
        dl = DataLoader(
            ds,
            batch_size=ds_params.get("bs", 1),
            shuffle=False,
            num_workers=ds_params.get("num_workers", 0),
            collate_fn=ds.mixer,
        )

    return dl, {"length": len(ds), "num_classes": ds_params.get("n_classes", 10)}
