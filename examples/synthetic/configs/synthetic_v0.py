"""The smallest complete gatle-ignite task, on synthetic data."""

from pathlib import Path

from examples.synthetic.configs.base_utils import ckpt_dir
from gatle_ignite import base_config

IN_DIM = 64
N_CLASSES = 10


def get_config():
    cfg = base_config()
    cfg.name = Path(__file__).stem
    cfg.project_name = "gatle-synthetic"
    cfg.save_dir = ckpt_dir(cfg.name)

    cfg.main_runner = "examples.synthetic.trainer.synthetic_trainer"
    cfg.model_name = "examples.synthetic.models.mlp"
    cfg.model_params = {"in_dim": IN_DIM, "hidden": 128, "n_classes": N_CLASSES}

    common = {"in_dim": IN_DIM, "n_classes": N_CLASSES, "bs": 64, "num_workers": 0}
    cfg.train_ds_name = "examples.synthetic.dataloaders.synthetic_dataset"
    cfg.train_ds_params = {**common, "n": 2048, "seed": 0, "shuffle": True, "drop_last": True}
    cfg.valid_ds_name = "examples.synthetic.dataloaders.synthetic_dataset"
    cfg.valid_ds_params = {**common, "n": 512, "seed": 0, "shuffle": False}

    cfg.criterion_name = "gatle_ignite.losses.composite"
    cfg.criterion_params = {
        "dict_of_loss_params": {
            "ce": {
                "cls_name": "examples.synthetic.losses.loss_functions.cross_entropy",
                "loss_params": {"src_name": "logits", "tgt_name": ("targets", "labels")},
                "weight": 1.0,
            }
        }
    }

    cfg.optimizer_name = "gatle_ignite.optimizers.adamw"
    cfg.optimizer_params = {"lr": 1e-3, "weight_decay": 0.01}
    cfg.lr_scheduler = "gatle_ignite.schedulers.warmup_cosine"
    cfg.lr_scheduler_params = {"warmup_epochs": 1}
    cfg.max_epochs = 15

    cfg.val_metrics = {
        "acc": {
            "cls_name": "examples.synthetic.metrics.accuracy",
            "params": {"src_name": "logits", "tgt_name": ("targets", "labels")},
        }
    }
    cfg.score_name = "valid/acc"
    cfg.score_factor = 1

    cfg.logger_name = ["text"]
    cfg.amp_dtype = "fp32"
    return cfg
