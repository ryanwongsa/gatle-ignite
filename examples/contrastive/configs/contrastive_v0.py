"""SimCLR-style contrastive learning: a loss with no targets, scored by a kNN probe."""

from examples.contrastive.configs.base_utils import ckpt_dir
from gatle_ignite import base_config

OBS_DIM = 64
N_CLASSES = 10

# One task for every split. At style_scale 4.0 a kNN on raw inputs scores ~0.40 against a
# content oracle's 1.00; at 1.5 raw inputs already score 1.000 and the run shows nothing.
TASK = {
    "n_classes": N_CLASSES,
    "d_content": 16,
    "d_style": 16,
    "obs_dim": OBS_DIM,
    "proto_scale": 1.0,
    "content_sigma": 0.5,
    "style_scale": 4.0,
    "num_workers": 0,
}


def get_config():
    cfg = base_config()
    cfg.name = "contrastive_ntxent"
    cfg.project_name = "gatle-contrastive"
    cfg.save_dir = ckpt_dir(cfg.name)

    cfg.main_runner = "examples.contrastive.trainer.contrastive_trainer"
    cfg.model_name = "examples.contrastive.models.encoder"
    cfg.model_params = {"in_dim": OBS_DIM, "hidden": 256, "feat_dim": 128, "proj_dim": 32}

    cfg.train_ds_name = "examples.contrastive.dataloaders.twoview_dataset"
    cfg.train_ds_params = {
        **TASK,
        "n": 4096,
        "bs": 256,
        "seed": 0,
        "shuffle": True,
        # Required: the number of negatives IS the difficulty, and a short batch changes it.
        "drop_last": True,
        # Fresh styles every __getitem__: this task's augmentation, train split only.
        "stochastic_views": True,
    }

    # Query set for the probe. Labels here are never trained on.
    cfg.valid_ds_name = "examples.contrastive.dataloaders.twoview_dataset"
    cfg.valid_ds_params = {**TASK, "n": 1024, "bs": 256, "seed": 1, "shuffle": False}

    # The kNN bank, read by our trainer. shard=False: every rank encodes the whole bank.
    cfg.bank_ds_name = "examples.contrastive.dataloaders.twoview_dataset"
    cfg.bank_ds_params = {**TASK, "n": 2048, "bs": 256, "seed": 2, "shard": False}

    cfg.criterion_name = "gatle_ignite.losses.composite"
    cfg.criterion_params = {
        "dict_of_loss_params": {
            "ntxent": {
                "cls_name": "examples.contrastive.losses.loss_functions.ntxent",
                # No tgt_name. There is nothing in `targets` for this loss to read.
                "loss_params": {"src_a": "z1", "src_b": "z2", "temperature": 0.2},
                "weight": 1.0,
            }
        }
    }

    cfg.optimizer_name = "gatle_ignite.optimizers.adamw"
    cfg.optimizer_params = {"lr": 3e-3, "weight_decay": 1e-4}
    cfg.lr_scheduler = "gatle_ignite.schedulers.warmup_cosine"
    cfg.lr_scheduler_params = {"warmup_epochs": 1}
    # The probe plateaus near epoch 5 while the loss keeps falling: the loss is not the goal.
    cfg.max_epochs = 10
    cfg.every_val = 1

    # Empty: the only eval metric is built imperatively in the trainer.
    cfg.val_metrics = {}
    cfg.knn_params = {"src_name": "h1", "k": 20, "temperature": 0.1}

    cfg.score_name = "valid/knn"
    cfg.score_factor = 1

    cfg.logger_name = ["text"]
    cfg.amp_dtype = "fp32"
    return cfg
