"""A diverged run stops, and nothing it saves afterwards comes from the diverged model.

Every test drives a real `fit()`: what matters is which handlers still fire after the stop,
and that differs between ignite releases.
"""

import sys
import warnings
from pathlib import Path

import pytest
from examples.synthetic.configs.synthetic_v0 import get_config

from gatle_ignite.cli.launch import run_task
from gatle_ignite.dispatch import ConfigError
from gatle_ignite.utils.checkpoints import resolve_eval_checkpoint

# 0.5 at the first evaluation, NaN from the second.
LATE_NAN_METRIC = """
from ignite.metrics import Metric


class _LateNaN(Metric):
    calls = 0

    def reset(self): pass
    def update(self, output): pass

    def compute(self):
        type(self).calls += 1
        return 0.5 if type(self).calls == 1 else float("nan")


def get_metric(engine_type, info, **params):
    return _LateNaN(output_transform=lambda output: None)
"""

NAN_METRIC = """
from ignite.metrics import Metric


class _NaN(Metric):
    def reset(self): pass
    def update(self, output): pass
    def compute(self): return float("nan")


def get_metric(engine_type, info, **params):
    return _NaN(output_transform=lambda output: None)
"""

# A loss term that is NaN whenever it is active; `start_iteration` gates when that begins.
NAN_LOSS = """
import torch.nn as nn


class Loss(nn.Module):
    def forward(self, y_pred, target, iteration=None):
        return y_pred["logits"].sum() * float("nan")
"""


@pytest.fixture
def write_module(tmp_path, monkeypatch):

    def _write(name, source):
        (tmp_path / f"{name}.py").write_text(source)
        monkeypatch.syspath_prepend(str(tmp_path))
        sys.modules.pop(name, None)
        return name

    return _write


@pytest.fixture
def cfg(tmp_path):
    cfg = get_config()
    cfg.save_dir = str(tmp_path / "ck")
    cfg.max_epochs = 4
    cfg.n_saved = None  # keep every file, so a save from a diverged epoch would show
    cfg.logger_name = []
    cfg.train_ds_params["n"] = 256  # 4 iterations per epoch
    cfg.valid_ds_params["n"] = 64
    return cfg


def _score_on(cfg, module):
    cfg.val_metrics = {"late": {"cls_name": module, "params": {}}}
    cfg.score_name = "valid/late"


def _names(cfg, prefix):
    return sorted(p.name for p in Path(cfg.save_dir).glob(f"{prefix}*.pt"))


def test_a_nan_score_stops_the_run_and_is_never_saved(cfg, write_module):
    _score_on(cfg, write_module("late_nan_metric", LATE_NAN_METRIC))
    task = run_task(0, cfg)

    assert task.engines["trainer"].state.epoch == 2
    assert _names(cfg, "valid_best") == ["valid_best_result_checkpoint_1_0.5000.pt"]
    assert _names(cfg, "latest") == ["latest_epoch_checkpoint_1.pt"]
    best = resolve_eval_checkpoint(cfg.save_dir, "best", prefix="valid_")
    assert best.name == "valid_best_result_checkpoint_1_0.5000.pt"


def test_a_run_never_finite_leaves_no_best_to_evaluate(cfg, write_module):
    _score_on(cfg, write_module("nan_metric", NAN_METRIC))
    run_task(0, cfg)

    eval_cfg = get_config()
    eval_cfg.save_dir = cfg.save_dir
    eval_cfg.run = True
    eval_cfg.load_from_ckpt = "best"
    eval_cfg.logger_name = []
    with pytest.raises(ConfigError, match="no 'best' checkpoint found"):
        run_task(0, eval_cfg)


def test_a_nan_loss_stops_mid_epoch_and_writes_no_latest_for_it(cfg, write_module):
    cfg.criterion_params["dict_of_loss_params"]["nan"] = {
        "cls_name": write_module("nan_loss", NAN_LOSS),
        "weight": 1.0,
        "start_iteration": 6,  # the second iteration of epoch 2
    }
    task = run_task(0, cfg)

    state = task.engines["trainer"].state
    assert (state.epoch, state.iteration) == (2, 6)
    assert _names(cfg, "latest") == ["latest_epoch_checkpoint_1.pt"]


def test_the_message_names_the_trainers_epoch(cfg, write_module):
    _score_on(cfg, write_module("late_nan_metric", LATE_NAN_METRIC))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        run_task(0, cfg)

    messages = [str(w.message) for w in caught if "valid/late is nan" in str(w.message)]
    assert messages, "no warning about the non-finite score"
    assert "at epoch 2" in messages[0]
