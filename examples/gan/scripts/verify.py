"""Prove the GAN learned rather than merely ran: its losses are adversarial and prove nothing.

TRAINED must beat UNTRAINED (the positive control, same seed) by MIN_IMPROVEMENT, and land
within MAX_FLOOR_MULTIPLE of the real-vs-real FLOOR. valid/swd is stochastic in z, so it
differs a little from the score in the checkpoint's filename.
"""

import torch

from examples.gan.configs.gan_v0 import get_config
from examples.gan.dataloaders.ring_dataset import sample_real
from examples.gan.metrics.distribution import _sliced_wasserstein
from examples.gan.trainer.gan_trainer import Trainer

# Well below the observed ~14x, so it fails on a broken loop and not on jitter.
MIN_IMPROVEMENT = 5.0

# Catches "it improved, but nowhere near the real distribution".
MAX_FLOOR_MULTIPLE = 12.0


def evaluate(load_checkpoint):
    """cfg.run attaches no checkpoint handlers, so scoring random weights cannot write 'best'."""
    cfg = get_config()
    cfg.run = True
    cfg.load_from_ckpt = "best"
    task = Trainer(0, cfg)
    # Unloaded, cfg.seed makes these exactly the weights training started from.
    return task.evaluate(load_checkpoint=load_checkpoint)


def main():
    torch.set_num_threads(1)

    # The floor: the eval set (seed 99) against an independent draw from the same truth.
    floor = _sliced_wasserstein(sample_real(4096, seed=99), sample_real(4096, seed=1234)).item()

    print("\n=== UNTRAINED (positive control) ===")
    untrained = evaluate(load_checkpoint=False)
    print("\n=== TRAINED (best checkpoint) ===")
    trained = evaluate(load_checkpoint=True)

    u_swd, t_swd = untrained["valid/swd"], trained["valid/swd"]
    print("\n" + "=" * 58)
    print(f"{'FLOOR     real vs real (sampling noise)':<46} swd = {floor:.4f}")
    print(f"{'UNTRAINED generator at init':<46} swd = {u_swd:.4f}")
    print(f"{'TRAINED   generator, best checkpoint':<46} swd = {t_swd:.4f}")
    print(f"{'':<46} moments: {untrained['valid/moments']:.4f} -> {trained['valid/moments']:.4f}")
    print("=" * 58)
    print(f"improvement over untrained: {u_swd / t_swd:.1f}x   (required: >{MIN_IMPROVEMENT}x)")
    print(f"distance above floor:       {t_swd / floor:.1f}x   (required: <{MAX_FLOOR_MULTIPLE}x)")

    assert u_swd / t_swd > MIN_IMPROVEMENT, (
        f"generator did not meaningfully beat its own initialisation: untrained swd "
        f"{u_swd:.4f} vs trained {t_swd:.4f} ({u_swd / t_swd:.1f}x, need "
        f"> {MIN_IMPROVEMENT}x). The training loop is not updating the generator."
    )
    assert t_swd / floor < MAX_FLOOR_MULTIPLE, (
        f"generator improved but is still far from the real distribution: trained swd "
        f"{t_swd:.4f} is {t_swd / floor:.1f}x the sampling-noise floor {floor:.4f} "
        f"(need < {MAX_FLOOR_MULTIPLE}x)."
    )
    print("\nPASS: the generator moved from noise to the real distribution.")


if __name__ == "__main__":
    main()
