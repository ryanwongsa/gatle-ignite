"""Proof that the trained model generates the ring, not just that a loss fell.

Samples are scored beside a real-vs-real floor, an N(0,I) prior that fools mode_tv, and an
untrained model: the positive control, on the same code path.
"""

import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[3]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from examples.diffusion.configs.diffusion_v0 import get_config  # noqa: E402
from examples.diffusion.dataloaders.ring_dataset import get_ds  # noqa: E402
from examples.diffusion.metrics.ring_stats import STATS, RingDistributionStats  # noqa: E402
from examples.diffusion.trainer.diffusion_trainer import Trainer  # noqa: E402


def eval_cfg():
    cfg = get_config()
    cfg.run = True  # inference mode: no checkpoint handlers, evaluate() will load weights
    cfg.logger_name = []
    return cfg


def run(load_checkpoint):
    trainer = Trainer(0, eval_cfg())
    return trainer.evaluate(load_checkpoint=load_checkpoint)


def _score(fake, real, centers):
    out = {}
    for stat in STATS:
        m = RingDistributionStats(centers=centers, stat=stat)
        m.reset()
        m.update((fake, real))
        out[f"valid/{stat}"] = m.compute()
    return out


def _reference_points():
    cfg = get_config()
    _, info = get_ds(dict(cfg.valid_ds_params))
    n = cfg.valid_ds_params["n"]
    gen = torch.Generator().manual_seed(1)
    from examples.diffusion.dataloaders.ring_dataset import mode_centers

    centers = mode_centers(info["n_modes"], info["radius"])
    which = torch.randint(0, info["n_modes"], (2 * n,), generator=gen)
    pts = centers[which] + info["std"] * torch.randn(2 * n, 2, generator=gen)
    return centers, pts[:n], pts[n:], gen


def baselines():
    """The real-vs-real floor, and an N(0, I) prior: what a sampler that never denoises gives.

    The prior fools mode_tv; mode_dist and radius_error are what catch it.
    """
    centers, a, b, gen = _reference_points()
    prior = torch.randn(a.shape, generator=gen)
    return _score(a, b, centers), _score(prior, b, centers)


if __name__ == "__main__":
    torch.set_num_threads(1)
    floor, prior = baselines()
    print("\n--- untrained (positive control) ---")
    untrained = run(load_checkpoint=False)
    print("\n--- trained (best checkpoint) ---")
    trained = run(load_checkpoint=True)

    print(
        f"\n{'metric':<22}{'real vs real':>14}{'prior N(0,I)':>14}{'UNTRAINED':>14}{'TRAINED':>14}"
    )
    for stat in STATS:
        key = f"valid/{stat}"
        print(
            f"{key:<22}{floor[key]:>14.4f}{prior[key]:>14.4f}"
            f"{untrained[key]:>14.4f}{trained[key]:>14.4f}"
        )

    # Beat both negatives on the stats each of them can fool.
    ok = (
        trained["valid/mode_tv"] < 0.5 * untrained["valid/mode_tv"]
        and trained["valid/mode_dist"] < 0.5 * untrained["valid/mode_dist"]
        and trained["valid/mode_dist"] < 0.5 * prior["valid/mode_dist"]
        and trained["valid/radius_error"] < 0.5 * prior["valid/radius_error"]
    )
    print(
        "\nPASS: sampling learned the distribution"
        if ok
        else "\nFAIL: samples are no better than an untrained model"
    )
    sys.exit(0 if ok else 1)
