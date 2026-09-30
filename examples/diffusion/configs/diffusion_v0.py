"""A DDPM on eight Gaussians on a ring: eval samples points and scores them against real data."""

from pathlib import Path

from examples.diffusion.configs.base_utils import ckpt_dir
from gatle_ignite import base_config

NUM_STEPS = 200


def get_config():
    cfg = base_config()
    cfg.name = Path(__file__).stem
    cfg.project_name = "gatle-diffusion"
    cfg.save_dir = ckpt_dir(cfg.name)

    cfg.main_runner = "examples.diffusion.trainer.diffusion_trainer"
    cfg.model_name = "examples.diffusion.models.denoiser"
    cfg.model_params = {"dim": 2, "hidden": 256, "time_dim": 64, "n_layers": 3}

    # A project field: the trainer builds the noise schedule from it; the framework ignores it.
    cfg.diffusion = {"num_steps": NUM_STEPS, "torch_threads": 1}

    cfg.train_ds_name = "examples.diffusion.dataloaders.ring_dataset"
    cfg.train_ds_params = {
        "n": 4096,
        "seed": 0,
        "bs": 256,
        "num_workers": 0,
        "shuffle": True,
        "drop_last": True,
    }
    # Held-out real data: the reference distribution the samples are scored against.
    cfg.valid_ds_name = "examples.diffusion.dataloaders.ring_dataset"
    cfg.valid_ds_params = {"n": 1024, "seed": 7, "bs": 1024, "num_workers": 0, "shuffle": False}

    cfg.criterion_name = "gatle_ignite.losses.composite"
    cfg.criterion_params = {
        "dict_of_loss_params": {
            "eps": {
                "cls_name": "examples.diffusion.losses.loss_functions.mse",
                # The target is the noise drawn inside prep_batch, not dataset content.
                "loss_params": {"src_name": "noise_pred", "tgt_name": ("targets", "noise")},
                "weight": 1.0,
            }
        }
    }

    cfg.optimizer_name = "gatle_ignite.optimizers.adamw"
    cfg.optimizer_params = {"lr": 2e-3, "weight_decay": 0.0}
    cfg.lr_scheduler = "gatle_ignite.schedulers.warmup_cosine"
    cfg.lr_scheduler_params = {"warmup_epochs": 2}
    cfg.max_epochs = 60

    _stats = {
        "cls_name": "examples.diffusion.metrics.ring_stats",
        "params": {"src_name": "samples", "tgt_name": ("targets", "x0")},
    }
    cfg.val_metrics = {
        "mode_tv": {**_stats, "params": {**_stats["params"], "stat": "mode_tv"}},
        "mode_dist": {**_stats, "params": {**_stats["params"], "stat": "mode_dist"}},
        "radius_error": {**_stats, "params": {**_stats["params"], "stat": "radius_error"}},
    }
    # Sampling 1024 points is 200 forward passes, so don't do it every epoch.
    cfg.every_val = 20
    # Score on mode_dist, NOT mode_tv: pure N(0,I) noise scores mode_tv 0.055 against real
    # data's 0.057 floor, while mode_dist separates them 0.19 vs 0.61.
    cfg.score_name = "valid/mode_dist"
    cfg.score_factor = -1  # lower is better; ignite keeps the maximum

    cfg.logger_name = ["text"]
    cfg.amp_dtype = "fp32"
    return cfg
