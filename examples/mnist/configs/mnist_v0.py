"""MNIST: torchvision augmentation, label smoothing, a test split. Needs the [examples] extra."""

from examples.mnist.configs.base_utils import ckpt_dir
from gatle_ignite import base_config


def get_config():
    cfg = base_config()
    cfg.name = "mnist_cnn"
    cfg.project_name = "gatle-mnist"
    cfg.save_dir = ckpt_dir(cfg.name)

    cfg.main_runner = "examples.mnist.trainer.mnist_trainer"
    cfg.model_name = "examples.mnist.models.cnn"
    cfg.model_params = {"n_classes": 10, "width": 32, "dropout": 0.25}

    cfg.aug_name = "examples.mnist.augmentation.mnist_aug"
    cfg.aug_params = {"degrees": 10, "translate": 0.1}

    common = {"root": "./data", "bs": 128, "num_workers": 2}
    cfg.train_ds_name = "examples.mnist.dataloaders.mnist_dataset"
    cfg.train_ds_params = {**common, "split": "train", "shuffle": True, "drop_last": True}
    # MNIST has no separate val split; the test set doubles as both here.
    cfg.valid_ds_name = "examples.mnist.dataloaders.mnist_dataset"
    cfg.valid_ds_params = {**common, "split": "test", "shuffle": False}
    cfg.test_ds_name = "examples.mnist.dataloaders.mnist_dataset"
    cfg.test_ds_params = {**common, "split": "test", "shuffle": False}

    cfg.criterion_name = "gatle_ignite.losses.composite"
    cfg.criterion_params = {
        "dict_of_loss_params": {
            "ce": {
                "cls_name": "examples.mnist.losses.loss_functions.cross_entropy",
                "loss_params": {
                    "src_name": "logits",
                    "tgt_name": ("targets", "labels"),
                    "label_smoothing": 0.1,
                },
                "weight": 1.0,
            }
        }
    }

    cfg.optimizer_name = "gatle_ignite.optimizers.adamw"
    cfg.optimizer_params = {"lr": 1e-3, "weight_decay": 0.01}
    cfg.lr_scheduler = "gatle_ignite.schedulers.warmup_cosine"
    cfg.lr_scheduler_params = {"warmup_epochs": 1, "lr_scale_factor": 0.01}
    cfg.max_epochs = 5
    cfg.grad_clip_norm = 1.0

    metric = {
        "acc": {
            "cls_name": "examples.mnist.metrics.accuracy",
            "params": {"src_name": "logits", "tgt_name": ("targets", "labels")},
        }
    }
    cfg.val_metrics = metric
    cfg.tester_metrics = metric
    cfg.every_val = 1
    cfg.every_test = 5

    cfg.score_name, cfg.score_factor = "valid/acc", 1
    cfg.logger_name = ["text", "pbar"]  # add "wandb" once WANDB_API_KEY is set
    cfg.resume = True
    return cfg
