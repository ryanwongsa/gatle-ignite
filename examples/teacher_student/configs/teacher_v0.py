"""Distillation, stage 1: a wide teacher on a large training set."""

from examples.teacher_student.configs import base_utils as P
from gatle_ignite import base_config


def get_config():
    cfg = base_config()
    cfg.name = "teacher"
    cfg.project_name = "gatle-teacher-student"
    cfg.save_dir = str(P.TEACHER_DIR)

    cfg.main_runner = "examples.teacher_student.trainer.teacher_trainer"
    cfg.model_name = "examples.teacher_student.models.mlp"
    cfg.model_params = dict(P.TEACHER_PARAMS)

    cfg.train_ds_name = "examples.teacher_student.dataloaders.synthetic_dataset"
    cfg.train_ds_params = dict(P.TEACHER_TRAIN)
    cfg.valid_ds_name = "examples.teacher_student.dataloaders.synthetic_dataset"
    cfg.valid_ds_params = dict(P.VALID)

    cfg.criterion_name = "gatle_ignite.losses.composite"
    cfg.criterion_params = {
        "dict_of_loss_params": {
            "ce": {
                "cls_name": "examples.teacher_student.losses.loss_functions.cross_entropy",
                "loss_params": {"src_name": "logits", "tgt_name": ("targets", "labels")},
                "weight": 1.0,
            }
        }
    }

    cfg.optimizer_name = "gatle_ignite.optimizers.adamw"
    cfg.optimizer_params = {"lr": 3e-3, "weight_decay": 0.01}
    cfg.lr_scheduler = "gatle_ignite.schedulers.warmup_cosine"
    cfg.lr_scheduler_params = {"warmup_epochs": 1}
    cfg.max_epochs = 12

    cfg.val_metrics = {
        "acc": {
            "cls_name": "examples.teacher_student.metrics.accuracy",
            "params": {"src_name": "logits", "tgt_name": ("targets", "labels")},
        }
    }
    cfg.score_name = "valid/acc"
    cfg.score_factor = 1

    cfg.logger_name = ["text"]
    cfg.log_every = 10_000  # a 30s run does not need a progress bar
    cfg.amp_dtype = "fp32"
    return cfg
