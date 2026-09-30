"""gatle-ignite: a config-driven pytorch-ignite training framework."""

__version__ = "0.4.1"

from gatle_ignite.config.base import base_config
from gatle_ignite.data.helpers import build_dataloader, get_aug, get_dataset
from gatle_ignite.dispatch import ConfigError
from gatle_ignite.trainer.base import BaseTrainer
from gatle_ignite.trainer.specs import EngineSpec
from gatle_ignite.utils.batch import get_value, to_device
from gatle_ignite.utils.checkpoints import load_weights

__all__ = [
    "BaseTrainer",
    "ConfigError",
    "EngineSpec",
    "base_config",
    "build_dataloader",
    "get_aug",
    "get_dataset",
    "get_value",
    "load_weights",
    "to_device",
    "__version__",
]
