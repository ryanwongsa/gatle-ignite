"""Distance between generated and real distributions, since GAN losses cannot say it works.

swd: sliced Wasserstein distance, exact per 1-D slice. moments: error in per-dim mean and std.
Both use the whole epoch: a per-batch distance, averaged, is biased by batch size.
"""

import ignite.distributed as idist
import torch
from ignite.metrics import Metric
from ignite.metrics.metric import reinit__is_reduced

from gatle_ignite import get_value


def _sliced_wasserstein(fake, real, n_proj=256, n_quantiles=256, seed=0):
    directions = torch.randn(fake.shape[1], n_proj, generator=torch.Generator().manual_seed(seed))
    directions = directions / directions.norm(dim=0, keepdim=True)

    levels = torch.linspace(0.0, 1.0, n_quantiles)
    # Quantiles, not sorted pairs, so the two sets may differ in size.
    q_fake = torch.quantile((fake @ directions).float(), levels, dim=0)
    q_real = torch.quantile((real @ directions).float(), levels, dim=0)
    return (q_fake - q_real).abs().mean()


def _moment_error(fake, real):
    mean_err = (fake.mean(0) - real.mean(0)).abs().mean()
    std_err = (fake.std(0) - real.std(0)).abs().mean()
    return mean_err + std_err


class DistributionDistance(Metric):
    """Accumulate every sample of an epoch, then compare the two sets.

    Samples are all-gathered, not @sync_all_reduce'd: a distance is no mean of per-rank ones.
    """

    def __init__(self, stat="swd", output_transform=lambda x: x, device="cpu"):
        if stat not in ("swd", "moments"):
            raise ValueError(f"stat must be 'swd' or 'moments', got {stat!r}")
        self.stat = stat
        super().__init__(output_transform=output_transform, device=device)

    @reinit__is_reduced
    def reset(self):
        self._fake = []
        self._real = []
        super().reset()

    @reinit__is_reduced
    def update(self, output):
        fake, real = output
        self._fake.append(fake.detach().to(self._device))
        self._real.append(real.detach().to(self._device))

    def compute(self):
        if not self._fake:
            # A silent 0.0 would read as a perfect generator.
            raise ValueError(f"{type(self).__name__} needs at least one batch")
        fake, real = torch.cat(self._fake), torch.cat(self._real)
        if idist.get_world_size() > 1:
            fake, real = idist.all_gather(fake), idist.all_gather(real)
        if self.stat == "swd":
            return _sliced_wasserstein(fake, real).item()
        return _moment_error(fake, real).item()


def get_metric(
    engine_type, info, stat="swd", src_name="fake", tgt_name=("targets", "real"), **kwargs
):
    def output_transform(output):
        return get_value(output["y_pred"], src_name), get_value(output["target"], tgt_name)

    return DistributionDistance(stat=stat, output_transform=output_transform, **kwargs)
