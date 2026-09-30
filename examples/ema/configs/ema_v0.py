"""EMA: evaluate a shadow copy of the weights, with a live-weights engine as the control."""

from pathlib import Path

from examples.ema.configs.base_utils import ckpt_dir
from gatle_ignite import base_config

IN_DIM = 64
N_CLASSES = 10


def get_config():
    cfg = base_config()
    cfg.name = Path(__file__).stem
    cfg.project_name = "gatle-ema"
    cfg.save_dir = ckpt_dir(cfg.name)

    cfg.main_runner = "examples.ema.trainer.ema_trainer"
    cfg.model_name = "examples.ema.models.mlp"
    cfg.model_params = {"in_dim": IN_DIM, "hidden": 128, "n_classes": N_CLASSES}

    cfg.ema_decay = 0.99  # project field: read by our Trainer, ignored by the framework

    common = {"in_dim": IN_DIM, "n_classes": N_CLASSES, "num_workers": 0}
    cfg.train_ds_name = "examples.ema.dataloaders.synthetic_dataset"
    cfg.train_ds_params = {
        **common,
        "n": 2048,
        "bs": 16,  # small -> gradient noise
        "seed": 0,
        "shuffle": True,
        "drop_last": True,
        "label_noise": 0.2,
    }
    cfg.valid_ds_name = "examples.ema.dataloaders.synthetic_dataset"
    cfg.valid_ds_params = {**common, "n": 1024, "bs": 128, "seed": 7, "shuffle": False}

    cfg.criterion_name = "gatle_ignite.losses.composite"
    cfg.criterion_params = {
        "dict_of_loss_params": {
            "ce": {
                "cls_name": "examples.ema.losses.loss_functions.cross_entropy",
                "loss_params": {"src_name": "logits", "tgt_name": ("targets", "labels")},
                "weight": 1.0,
            }
        }
    }

    # Plain SGD at a high LR, no annealing: the point is that the endpoint is noisy.
    cfg.optimizer_name = "gatle_ignite.optimizers.sgd"
    cfg.optimizer_params = {"lr": 0.15, "momentum": 0.9, "weight_decay": 0.0}
    cfg.max_epochs = 6

    acc = {
        "cls_name": "examples.ema.metrics.accuracy",
        "params": {"src_name": "logits", "tgt_name": ("targets", "labels")},
    }
    cfg.val_metrics = {"acc": acc}  # evaluated with the EMA weights swapped in
    cfg.raw_metrics = {"acc": acc}  # the control engine: same data, live weights
    cfg.every_raw = 1  # a declared engine must say when it runs
    # No raw_ds_*: RAW_SPEC's ds_prefix="valid" shares the valid loader.

    cfg.score_name = "valid/acc"
    cfg.score_factor = 1

    cfg.logger_name = ["text"]
    cfg.amp_dtype = "fp32"
    return cfg
