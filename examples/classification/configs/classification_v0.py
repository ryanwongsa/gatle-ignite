"""Imbalanced classification: a class-balancing sampler, train-only augmentation, three engines."""

import os

from examples.classification.configs.base_utils import ckpt_dir
from examples.classification.dataloaders.data_utils.shapes import N_CLASSES
from examples.classification.dataloaders.shapes_dataset import DEPLOY_COUNTS
from gatle_ignite import base_config

# SHAPES_NO_SAMPLER=1 drops the sampler: rare classes fall to zero recall as plain acc rises.
BALANCED_SAMPLING = os.environ.get("SHAPES_NO_SAMPLER") != "1"


def get_config():
    cfg = base_config()
    cfg.name = "shapes_cnn" if BALANCED_SAMPLING else "shapes_cnn_nosampler"
    cfg.project_name = "gatle-classification"
    cfg.save_dir = ckpt_dir(cfg.name)

    cfg.main_runner = "examples.classification.trainer.classification_trainer"
    cfg.model_name = "examples.classification.models.shapes_cnn"
    cfg.model_params = {"n_classes": N_CLASSES, "width": 16, "dropout": 0.1}

    cfg.aug_name = "examples.classification.augmentation.shapes_aug"
    cfg.aug_params = {"p_flip": 0.5, "noise": 0.05, "max_shift": 2, "brightness": 0.2}

    cfg.train_ds_name = "examples.classification.dataloaders.shapes_dataset"
    cfg.train_ds_params = {
        "split": "train",
        "seed": 0,
        "bs": 64,
        "num_workers": 0,
        "drop_last": True,
        # No `shuffle`: the sampler owns the order, and torch allows only one of the two.
        "sampler_params": {
            "cls_name": "examples.classification.dataloaders.data_utils.balanced_sampler",
            "params": {"num_samples": 1140, "num_classes": N_CLASSES},
        },
    }
    if not BALANCED_SAMPLING:
        cfg.train_ds_params = {
            k: v for k, v in cfg.train_ds_params.items() if k != "sampler_params"
        }
        cfg.train_ds_params["shuffle"] = True

    eval_common = {"bs": 256, "num_workers": 0, "shuffle": False}  # never drop_last on eval
    cfg.valid_ds_name = "examples.classification.dataloaders.shapes_dataset"
    cfg.valid_ds_params = {**eval_common, "split": "valid", "seed": 1}
    # Test keeps the deployment skew, so test/acc and test/bal_acc disagree: that gap is the result.
    cfg.test_ds_name = "examples.classification.dataloaders.shapes_dataset"
    cfg.test_ds_params = {**eval_common, "split": "test", "seed": 2, "class_counts": DEPLOY_COUNTS}

    cfg.criterion_name = "gatle_ignite.losses.composite"
    cfg.criterion_params = {
        "dict_of_loss_params": {
            "ce": {
                "cls_name": "examples.classification.losses.loss_functions.cross_entropy",
                "loss_params": {
                    "src_name": "logits",
                    "tgt_name": ("targets", "labels"),
                    "label_smoothing": 0.05,
                },
                "weight": 1.0,
            }
        }
    }

    cfg.optimizer_name = "gatle_ignite.optimizers.adamw"
    cfg.optimizer_params = {"lr": 3e-3, "weight_decay": 0.01}
    cfg.lr_scheduler = "gatle_ignite.schedulers.warmup_cosine"
    cfg.lr_scheduler_params = {"warmup_epochs": 1, "lr_scale_factor": 0.01}
    cfg.max_epochs = 10
    cfg.grad_clip_norm = 1.0

    # Plain acc flatters a model that gave up on the rare classes, so score on balanced acc.
    metrics = {
        "acc": {
            "cls_name": "examples.classification.metrics.accuracy",
            "params": {"src_name": "logits", "tgt_name": ("targets", "labels")},
        },
        "top2": {
            "cls_name": "examples.classification.metrics.topk",
            "params": {"src_name": "logits", "tgt_name": ("targets", "labels"), "k": 2},
        },
        "bal_acc": {
            "cls_name": "examples.classification.metrics.per_class",
            "params": {"src_name": "logits", "tgt_name": ("targets", "labels"), "per_class": True},
        },
    }
    cfg.val_metrics = metrics
    cfg.tester_metrics = metrics
    cfg.every_val = 1
    cfg.every_test = 5  # the tester is expensive in real projects; run it on a cadence

    cfg.score_name, cfg.score_factor = "valid/bal_acc", 1
    cfg.tester_score_name, cfg.tester_score_factor = "test/bal_acc", 1

    cfg.logger_name = ["text"]
    cfg.amp_dtype = "fp32"  # CPU
    return cfg
