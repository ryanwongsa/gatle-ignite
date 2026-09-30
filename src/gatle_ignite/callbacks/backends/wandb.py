"""Weights & Biases backend.

The API key comes from WANDB_API_KEY only, never from the config.
"""

import json

from gatle_ignite.callbacks.logging import Backend as BaseBackend
from gatle_ignite.dispatch import ConfigError


class Backend(BaseBackend):
    def __init__(self, cfg):
        super().__init__(cfg)
        try:
            import wandb
        except ImportError as e:
            raise ConfigError(
                "logger_name includes 'wandb' but wandb is not installed. "
                "Install 'gatle-ignite[wandb]' (or '.[wandb]'), or drop 'wandb' from logger_name."
            ) from e
        self._wandb = wandb

        config = (
            json.loads(cfg.to_json_best_effort())
            if hasattr(cfg, "to_json_best_effort")
            else dict(cfg)
        )
        self.run = wandb.init(
            project=cfg.project_name,
            entity=cfg.get("wandb_entity", None),  # None -> the user's default entity
            id=cfg.name,
            name=cfg.name,
            config=config,
            tags=list(cfg.tags),
            resume="allow",
            save_code=False,
            allow_val_change=True,
            settings=wandb.Settings(init_timeout=600),
        )

    def log(self, metrics, step=None, epoch=None):
        payload = dict(metrics)
        if epoch is not None:
            payload["epoch"] = epoch
        self.run.log(payload)

    def watch(self, model):
        self.run.watch(model)

    def finish(self, failed=False):
        self.run.finish(exit_code=1 if failed else 0)
