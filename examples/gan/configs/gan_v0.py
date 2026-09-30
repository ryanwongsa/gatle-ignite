"""A GAN: two nets in one Module, two optimizers in one object, alternating updates."""

import torch

from examples.gan.configs.base_utils import ckpt_dir
from gatle_ignite import base_config

Z_DIM = 8
DATA_DIM = 2


def get_config():
    # ~9k-parameter nets: 32 iterations measured 0.29s on 1 thread against 2.7s on 4.
    # Delete this line for a real model.
    torch.set_num_threads(1)

    cfg = base_config()
    cfg.name = "gan_2d"
    cfg.project_name = "gatle-gan"
    cfg.save_dir = ckpt_dir("gan_2d")

    cfg.main_runner = "examples.gan.trainer.gan_trainer"
    cfg.model_name = "examples.gan.models.gan"
    cfg.model_params = {"z_dim": Z_DIM, "hidden": 64, "data_dim": DATA_DIM}

    cfg.train_ds_name = "examples.gan.dataloaders.ring_dataset"
    cfg.train_ds_params = {
        "n": 8192,
        "seed": 0,
        "bs": 256,
        "num_workers": 0,
        "shuffle": True,
        "drop_last": True,
    }
    cfg.valid_ds_name = "examples.gan.dataloaders.ring_dataset"
    # A different seed, or the score would measure memorisation as fit.
    cfg.valid_ds_params = {"n": 4096, "seed": 99, "bs": 1024, "num_workers": 0, "shuffle": False}

    # Not the composite: a GAN has no weighted sum of terms.
    cfg.criterion_name = "examples.gan.losses.gan_loss"
    cfg.criterion_params = {"label_smoothing": 0.1}

    # Not a builtin: this builds TWO optimizers.
    cfg.optimizer_name = "examples.gan.optimizer.dual_optimizer"
    cfg.optimizer_params = {"gen_lr": 2e-3, "disc_lr": 1e-3, "betas": (0.5, 0.999)}

    # No LR schedule: one schedule over both nets' param_groups is not what a GAN wants.

    cfg.max_epochs = 20
    cfg.every_val = 5

    cfg.val_metrics = {
        "swd": {"cls_name": "examples.gan.metrics.distribution", "params": {"stat": "swd"}},
        "moments": {"cls_name": "examples.gan.metrics.distribution", "params": {"stat": "moments"}},
    }
    # Lower is better, and ignite keeps the maximum: without -1, "best" is the worst generator.
    cfg.score_name = "valid/swd"
    cfg.score_factor = -1

    cfg.logger_name = ["text"]
    cfg.amp_dtype = "fp32"
    cfg.log_every = 100
    return cfg
