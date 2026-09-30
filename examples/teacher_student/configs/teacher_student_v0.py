"""Distillation, stage 2: train a student against a second, frozen teacher model."""

from examples.teacher_student.configs import base_utils as P
from gatle_ignite import base_config

TEMPERATURE = 4.0


def get_config():
    cfg = base_config()
    cfg.name = "student_kd"
    cfg.project_name = "gatle-teacher-student"
    cfg.save_dir = str(P.STUDENT_DIR)

    cfg.main_runner = "examples.teacher_student.trainer.student_trainer"

    # The student: the one model the framework optimizes and checkpoints.
    cfg.model_name = "examples.teacher_student.models.mlp"
    cfg.model_params = dict(P.STUDENT_PARAMS)

    # The teacher: project fields our trainer reads. Not model_checkpoint_dir: that loads the student.
    cfg.teacher_model_name = "examples.teacher_student.models.mlp"
    cfg.teacher_model_params = dict(P.TEACHER_PARAMS)
    cfg.teacher_weights = P.teacher_ckpt()
    cfg.teacher_autotrain = True  # train stage 1 on demand if the checkpoint is absent

    cfg.train_ds_name = "examples.teacher_student.dataloaders.synthetic_dataset"
    cfg.train_ds_params = dict(P.STUDENT_TRAIN)
    cfg.valid_ds_name = "examples.teacher_student.dataloaders.synthetic_dataset"
    cfg.valid_ds_params = dict(P.VALID)

    cfg.criterion_name = "gatle_ignite.losses.composite"
    cfg.criterion_params = {
        "dict_of_loss_params": {
            "ce": {
                "cls_name": "examples.teacher_student.losses.loss_functions.cross_entropy",
                "loss_params": {"src_name": "logits", "tgt_name": ("targets", "labels")},
                "weight": 0.5,
            },
            "kd": {
                "cls_name": "examples.teacher_student.losses.loss_functions.kd_loss",
                # The teacher's logits, which prep_batch puts in targets.
                "loss_params": {
                    "src_name": "logits",
                    "tgt_name": ("targets", "teacher_logits"),
                    "temperature": TEMPERATURE,
                },
                "weight": 0.5,
            },
        }
    }

    cfg.optimizer_name = "gatle_ignite.optimizers.adamw"
    cfg.optimizer_params = {"lr": 3e-3, "weight_decay": 0.01}
    cfg.lr_scheduler = "gatle_ignite.schedulers.warmup_cosine"
    cfg.lr_scheduler_params = {"warmup_epochs": 2}
    cfg.max_epochs = 60

    cfg.val_metrics = {
        "acc": {
            "cls_name": "examples.teacher_student.metrics.accuracy",
            "params": {"src_name": "logits", "tgt_name": ("targets", "labels")},
        },
        # Agreement with the teacher's argmax on held-out data: what tells KD from a control.
        "teacher_agreement": {
            "cls_name": "examples.teacher_student.metrics.agreement",
            "params": {"src_name": "logits"},
        },
    }
    cfg.score_name = "valid/acc"
    cfg.score_factor = 1

    cfg.logger_name = ["text"]
    cfg.log_every = 10_000
    cfg.amp_dtype = "fp32"
    return cfg
