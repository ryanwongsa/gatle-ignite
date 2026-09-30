"""Scheduler wiring: every lr_scheduler_params value must reach get_scheduler."""

import pytest
import torch
from examples.synthetic.configs.synthetic_v0 import get_config
from ignite.engine import Engine

from gatle_ignite import base_config
from gatle_ignite.dispatch import ConfigError, import_entrypoint
from gatle_ignite.schedulers import prep_scheduler
from gatle_ignite.utils.checkpoints import find_latest_checkpoint

STEPS_PER_EPOCH = 10


@pytest.fixture
def pieces():
    cfg = base_config()
    cfg.max_epochs = 10
    cfg.optimizer_params = {"lr": 1e-3}
    model = torch.nn.Linear(4, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    return cfg, list(range(STEPS_PER_EPOCH)), optimizer


def _lr_curve(scheduler, optimizer, steps):
    curve = []
    for _ in range(steps):
        scheduler(None)
        curve.append(optimizer.param_groups[0]["lr"])
    return curve


@pytest.mark.parametrize("warmup_epochs", [1, 3])
def test_warmup_epochs_actually_controls_the_warmup(pieces, warmup_epochs):
    cfg, dl, optimizer = pieces
    cfg.lr_scheduler = "gatle_ignite.schedulers.warmup_cosine"
    cfg.lr_scheduler_params = {"warmup_epochs": warmup_epochs}

    scheduler, _, _ = prep_scheduler(
        cfg.lr_scheduler,
        cfg,
        dl,
        optimizer,
        {"trainer": Engine(lambda e, b: None), "evaluator": None},
    )
    curve = _lr_curve(scheduler, optimizer, STEPS_PER_EPOCH * cfg.max_epochs)
    peak = curve.index(max(curve)) + 1
    assert peak == warmup_epochs * STEPS_PER_EPOCH


def test_lr_scale_factor_sets_the_floor(pieces):
    cfg, dl, optimizer = pieces
    cfg.lr_scheduler = "gatle_ignite.schedulers.warmup_cosine"
    cfg.lr_scheduler_params = {"warmup_epochs": 1, "lr_scale_factor": 0.5}

    scheduler, _, _ = prep_scheduler(
        cfg.lr_scheduler,
        cfg,
        dl,
        optimizer,
        {"trainer": Engine(lambda e, b: None), "evaluator": None},
    )
    curve = _lr_curve(scheduler, optimizer, STEPS_PER_EPOCH * cfg.max_epochs)
    assert curve[-1] == pytest.approx(1e-3 * 0.5, rel=0.05)


def test_step_scheduler_gamma_is_honoured(pieces):
    cfg, dl, optimizer = pieces
    cfg.lr_scheduler = "gatle_ignite.schedulers.step"
    cfg.lr_scheduler_params = {"step_size": 1, "gamma": 0.1}

    scheduler, engine_name, _ = prep_scheduler(
        cfg.lr_scheduler,
        cfg,
        dl,
        optimizer,
        {"trainer": Engine(lambda e, b: None), "evaluator": None},
    )
    assert engine_name == "trainer"
    scheduler(None)
    scheduler(None)
    assert optimizer.param_groups[0]["lr"] == pytest.approx(1e-4, rel=1e-3)


def test_unknown_scheduler_param_is_a_config_error_not_a_typeerror(pieces):
    cfg, dl, optimizer = pieces
    cfg.lr_scheduler = "gatle_ignite.schedulers.warmup_cosine"
    cfg.lr_scheduler_params = {"warmup_epochz": 2}  # typo
    with pytest.raises(ConfigError, match="lr_scheduler_params"):
        prep_scheduler(
            cfg.lr_scheduler,
            cfg,
            dl,
            optimizer,
            {"trainer": Engine(lambda e, b: None), "evaluator": None},
        )


def test_optimizer_params_without_lr_still_works(pieces):
    cfg, dl, optimizer = pieces
    cfg.optimizer_params = {"weight_decay": 0.01}
    cfg.lr_scheduler = "gatle_ignite.schedulers.warmup_cosine"
    scheduler, _, _ = prep_scheduler(
        cfg.lr_scheduler,
        cfg,
        dl,
        optimizer,
        {"trainer": Engine(lambda e, b: None), "evaluator": None},
    )
    assert scheduler is not None


def test_one_batch_epoch_does_not_break_warmup(pieces):
    """The documented smoke-run workflow (train_length=1) yields a 1-event warmup."""
    cfg, dl, optimizer = pieces
    cfg.train_length = 1
    cfg.max_epochs = 1
    cfg.lr_scheduler = "gatle_ignite.schedulers.warmup_cosine"
    scheduler, _, _ = prep_scheduler(
        cfg.lr_scheduler,
        cfg,
        dl,
        optimizer,
        {"trainer": Engine(lambda e, b: None), "evaluator": None},
    )
    assert scheduler is not None


def test_none_scheduler_attaches_nothing(pieces):
    cfg, dl, optimizer = pieces
    scheduler, _, _ = prep_scheduler(
        None,
        cfg,
        dl,
        optimizer,
        {"trainer": Engine(lambda e, b: None), "evaluator": None},
    )
    assert scheduler is None


def test_cosine_anneals_from_lr_to_end_value(pieces):
    cfg, dl, optimizer = pieces
    cfg.lr_scheduler = "gatle_ignite.schedulers.cosine"
    cfg.lr_scheduler_params = {"end_value": 1e-5}

    scheduler, _, _ = prep_scheduler(
        cfg.lr_scheduler,
        cfg,
        dl,
        optimizer,
        {"trainer": Engine(lambda e, b: None), "evaluator": None},
    )
    curve = _lr_curve(scheduler, optimizer, STEPS_PER_EPOCH * cfg.max_epochs)
    assert curve[0] == pytest.approx(1e-3)
    assert curve[-1] == pytest.approx(1e-5, rel=0.05)


def test_plateau_trains_with_checkpoints_and_lowers_the_lr(tmp_path):
    cfg = get_config()
    cfg.save_dir = str(tmp_path)
    cfg.logger_name = []
    cfg.max_epochs = 2
    cfg.lr_scheduler = "gatle_ignite.schedulers.plateau"
    # "min" on an accuracy that rises means every evaluation after the first is "worse".
    cfg.lr_scheduler_params = {
        "metric_name": "valid/acc",
        "mode": "min",
        "patience": 0,
        "factor": 0.5,
    }
    task = import_entrypoint(cfg.main_runner, "Trainer", field="main_runner")(0, cfg)
    task.fit()

    state = torch.load(find_latest_checkpoint(cfg.save_dir), map_location="cpu")
    assert "scheduler" in state
    assert task.optimizer.param_groups[0]["lr"] < cfg.optimizer_params["lr"]


def test_the_latest_checkpoint_holds_the_plateau_state_after_its_evaluation(tmp_path):
    cfg = get_config()
    cfg.save_dir = str(tmp_path)
    cfg.logger_name = []
    cfg.max_epochs = 3
    cfg.lr_scheduler = "gatle_ignite.schedulers.plateau"
    # A patience longer than the run, so no LR drop resets the count.
    cfg.lr_scheduler_params = {"metric_name": "valid/acc", "mode": "min", "patience": 10}
    task = import_entrypoint(cfg.main_runner, "Trainer", field="main_runner")(0, cfg)
    task.fit()

    live = task.scheduler.state_dict()["scheduler"]["num_bad_epochs"]
    state = torch.load(find_latest_checkpoint(cfg.save_dir), map_location="cpu")
    assert live > 0, "no evaluation counted as bad, so this run cannot tell the orders apart"
    assert state["scheduler"]["scheduler"]["num_bad_epochs"] == live
