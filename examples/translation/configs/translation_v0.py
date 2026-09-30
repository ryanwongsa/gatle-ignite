"""Seq2seq translation: variable-length input, a padding collate, and a real decode to score."""

# Absolute imports only: the CLI loads a config by file path, so it has no parent package.
from examples.translation.configs.base_utils import ckpt_dir
from examples.translation.dataloaders.data_utils.vocab import MAX_TGT_LEN, TGT_VOCAB
from gatle_ignite import base_config

IN_DIM = 32


def get_config():
    cfg = base_config()
    cfg.name = "translation"
    cfg.project_name = "gatle-translation"
    cfg.save_dir = ckpt_dir(cfg.name)

    cfg.main_runner = "examples.translation.trainer.translation_trainer"
    cfg.model_name = "examples.translation.models.seq2seq"
    cfg.model_params = {
        "in_dim": IN_DIM,
        "d_model": 64,
        "nhead": 4,
        "num_encoder_layers": 2,
        "num_decoder_layers": 2,
        "dim_feedforward": 128,
        "dropout": 0.0,  # the task is noise-limited, not capacity-limited
        "tgt_vocab": TGT_VOCAB,
        "max_decode_len": MAX_TGT_LEN,
    }

    # A shared task_seed gives one toy language; each split's seed, different utterances.
    common = {"in_dim": IN_DIM, "task_seed": 1234, "num_workers": 0}
    cfg.train_ds_name = "examples.translation.dataloaders.toy_speech_dataset"
    cfg.train_ds_params = {
        **common,
        "n": 2048,
        "seed": 0,
        "bs": 64,
        "shuffle": True,
        "drop_last": True,
    }
    # Never drop_last on eval: a dropped tail is a quietly smaller test set.
    cfg.valid_ds_name = "examples.translation.dataloaders.toy_speech_dataset"
    cfg.valid_ds_params = {**common, "n": 256, "seed": 7, "bs": 128, "shuffle": False}

    cfg.criterion_name = "gatle_ignite.losses.composite"
    cfg.criterion_params = {
        "dict_of_loss_params": {
            "ce": {
                "cls_name": "examples.translation.losses.loss_functions.masked_ce",
                "loss_params": {
                    "src_name": "logits",
                    "tgt_name": ("targets", "tgt_out"),
                },
                "weight": 1.0,
            }
        }
    }

    cfg.optimizer_name = "gatle_ignite.optimizers.adamw"
    cfg.optimizer_params = {"lr": 2e-3, "weight_decay": 0.01}
    cfg.lr_scheduler = "gatle_ignite.schedulers.warmup_cosine"
    cfg.lr_scheduler_params = {"warmup_epochs": 1}
    cfg.max_epochs = 10  # converges to wer < 0.05 by ~epoch 5; 10 leaves headroom
    cfg.grad_clip_norm = 1.0

    # Both metrics score the autoregressive decode, never the teacher-forced logits.
    cfg.val_metrics = {
        "wer": {
            "cls_name": "examples.translation.metrics.edit_distance",
            "params": {
                "mode": "wer",
                "src_name": "pred_tokens",
                "tgt_name": ("targets", "tgt_out"),
            },
        },
        "exact": {
            "cls_name": "examples.translation.metrics.edit_distance",
            "params": {
                "mode": "exact",
                "src_name": "pred_tokens",
                "tgt_name": ("targets", "tgt_out"),
            },
        },
    }
    # Lower is better, and ignite keeps the maximum: without -1 it keeps the worst model.
    cfg.score_name = "valid/wer"
    cfg.score_factor = -1

    cfg.logger_name = ["text"]
    cfg.amp_dtype = "fp32"
    return cfg
