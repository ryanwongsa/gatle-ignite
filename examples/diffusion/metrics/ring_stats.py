"""Score generated samples against real ones: a noise-prediction loss falls either way.

mode_tv: TV distance between mode-occupancy histograms, 0 for the same and 1 for disjoint.
mode_dist: mean distance to the nearest mode centre; real data sits at ~std * sqrt(pi/2).
radius_error: |mean norm of samples - mean norm of real|, for a sampler with the wrong scale.
"""

import torch
from ignite.metrics import Metric
from ignite.metrics.metric import reinit__is_reduced, sync_all_reduce

from gatle_ignite import get_value

STATS = ("mode_tv", "mode_dist", "radius_error")


class RingDistributionStats(Metric):
    def __init__(self, centers, stat="mode_tv", output_transform=lambda x: x, device="cpu"):
        if stat not in STATS:
            raise ValueError(f"unknown stat {stat!r}; expected one of {STATS}")
        self.stat = stat
        self.centers = centers
        self.n_modes = centers.shape[0]
        super().__init__(output_transform=output_transform, device=device)

    @reinit__is_reduced
    def reset(self):
        z = lambda: torch.zeros(self.n_modes, dtype=torch.float64, device=self._device)  # noqa: E731
        self._hist_fake = z()
        self._hist_real = z()
        self._sum_mode_dist = torch.zeros(1, dtype=torch.float64, device=self._device)
        self._sum_radius_fake = torch.zeros(1, dtype=torch.float64, device=self._device)
        self._sum_radius_real = torch.zeros(1, dtype=torch.float64, device=self._device)
        self._n = torch.zeros(1, dtype=torch.float64, device=self._device)
        super().reset()

    def _assign(self, points):
        centers = self.centers.to(device=points.device, dtype=points.dtype)
        d = torch.cdist(points, centers)  # (B, n_modes)
        return d.min(dim=1)

    @reinit__is_reduced
    def update(self, output):
        fake, real = output
        fake, real = fake.detach().double(), real.detach().double()

        dist_fake, idx_fake = self._assign(fake)
        _, idx_real = self._assign(real)

        self._hist_fake += (
            torch.bincount(idx_fake, minlength=self.n_modes).double().to(self._device)
        )
        self._hist_real += (
            torch.bincount(idx_real, minlength=self.n_modes).double().to(self._device)
        )
        self._sum_mode_dist += dist_fake.sum().to(self._device)
        self._sum_radius_fake += fake.norm(dim=1).sum().to(self._device)
        self._sum_radius_real += real.norm(dim=1).sum().to(self._device)
        self._n += fake.shape[0]

    @sync_all_reduce(
        "_hist_fake", "_hist_real", "_sum_mode_dist", "_sum_radius_fake", "_sum_radius_real", "_n"
    )
    def compute(self):
        if self._n.item() == 0:
            # A silent 0.0 would read as a perfect score.
            raise ValueError("RingDistributionStats needs at least one sample")
        n = self._n.item()
        if self.stat == "mode_tv":
            p, q = self._hist_fake / n, self._hist_real / n
            return 0.5 * (p - q).abs().sum().item()
        if self.stat == "mode_dist":
            return (self._sum_mode_dist / n).item()
        return abs((self._sum_radius_fake / n).item() - (self._sum_radius_real / n).item())


def get_metric(
    engine_type,
    info,
    stat="mode_tv",
    src_name="samples",
    tgt_name=("targets", "x0"),
    **kwargs,
):
    centers = (info or {}).get("mode_centers")
    if centers is None:
        raise ValueError(
            "RingDistributionStats needs 'mode_centers' in the dataset info dict; "
            f"{engine_type} info has {sorted((info or {}).keys())}"
        )

    def output_transform(output):
        return get_value(output["y_pred"], src_name), get_value(output["target"], tgt_name)

    return RingDistributionStats(
        centers=centers, stat=stat, output_transform=output_transform, **kwargs
    )
