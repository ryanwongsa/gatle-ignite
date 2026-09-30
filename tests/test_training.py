"""End-to-end: the framework must actually train, checkpoint, and resume."""

import importlib
from pathlib import Path

import pytest
import torch
from examples.synthetic.configs.synthetic_v0 import get_config
from ignite.engine import Events

from gatle_ignite.cli.launch import run_task
from gatle_ignite.utils.checkpoints import find_best_checkpoint, find_latest_checkpoint


@pytest.fixture
def cfg(tmp_path):
    cfg = get_config()
    cfg.save_dir = str(tmp_path)
    cfg.max_epochs = 3
    cfg.logger_name = []
    cfg.train_ds_params["n"] = 256
    cfg.valid_ds_params["n"] = 128
    return cfg


def test_loss_components_are_named_from_crit_keys(cfg):
    task = run_task(0, cfg)
    losses = task.engines["trainer"].state.metrics
    assert "train/loss_avg" in losses
    assert "train/loss_ce_avg" in losses  # per-component, named from crit_keys


def test_it_actually_learns(cfg):
    """Backs the numbers docs/status.md quotes: loss 2.31 -> 0.59, accuracy 0.22 -> 0.93."""
    first, last = [], []
    # The example's own settings: the LR schedule derives from max_epochs.
    cfg.max_epochs = 15
    cfg.train_ds_params["n"] = 2048

    task = importlib.import_module(cfg.main_runner).Trainer(0, cfg)
    task.setup()
    trainer = task.engines["trainer"]

    @trainer.on(Events.EPOCH_COMPLETED)
    def record(engine):
        (first if engine.state.epoch == 1 else last).append(engine.state.metrics["train/loss_avg"])

    task.fit()

    acc = task.engines["evaluator"].state.metrics["valid/acc"]
    # Generous margins: this must fail on a broken loop, not on a different BLAS.
    assert last[-1] < 1.0, f"loss went {first[0]:.3f} -> {last[-1]:.3f}; docs claim ~0.59"
    assert last[-1] < first[0] * 0.5, f"loss barely moved: {first[0]:.3f} -> {last[-1]:.3f}"
    assert acc > 0.8, f"valid/acc {acc:.3f}; docs claim ~0.93 on a linearly separable task"


def test_writes_latest_and_best_checkpoints(cfg):
    run_task(0, cfg)
    assert find_latest_checkpoint(cfg.save_dir) is not None
    assert find_best_checkpoint(cfg.save_dir, prefix="valid_") is not None


def test_checkpoint_contains_the_expected_objects(cfg):
    run_task(0, cfg)
    state = torch.load(find_latest_checkpoint(cfg.save_dir), map_location="cpu", weights_only=False)
    assert {"model", "optimizer", "trainer"} <= set(state)


def test_resume_continues_from_the_saved_epoch(cfg):
    run_task(0, cfg)  # epochs 1-3

    cfg2 = get_config()
    cfg2.save_dir = cfg.save_dir
    cfg2.max_epochs = 5
    cfg2.resume = True
    cfg2.logger_name = []
    cfg2.train_ds_params["n"] = 256
    cfg2.valid_ds_params["n"] = 128
    task = run_task(0, cfg2)
    assert task.engines["trainer"].state.epoch == 5


def test_eval_mode_loads_a_checkpoint_without_training(cfg):
    run_task(0, cfg)

    eval_cfg = get_config()
    eval_cfg.save_dir = cfg.save_dir
    eval_cfg.run = True
    eval_cfg.load_from_ckpt = "best"
    eval_cfg.logger_name = []
    task = run_task(0, eval_cfg)

    # Engines and dataloaders are exposed, but no training happened.
    assert task.engines["trainer"].state.epoch == 0
    assert task.dls["valid"] is not None


def test_tester_engine_is_not_built_without_a_test_set(cfg):
    task = run_task(0, cfg)
    assert task.engines["tester"] is None


def test_example_trainer_stays_tiny():
    """A task's trainer is prep_batch, not boilerplate."""
    source = (
        Path(__file__).parent.parent / "examples" / "synthetic" / "trainer" / "synthetic_trainer.py"
    )
    assert len(source.read_text().strip().splitlines()) < 15


def test_eval_mode_never_writes_checkpoints(cfg):
    """Driving the evaluator in run-mode must not overwrite what it is evaluating."""
    run_task(0, cfg)
    best = find_best_checkpoint(cfg.save_dir, prefix="valid_")
    before = {p.name: p.stat().st_mtime for p in Path(cfg.save_dir).glob("*.pt")}

    eval_cfg = get_config()
    eval_cfg.save_dir = cfg.save_dir
    eval_cfg.run = True
    eval_cfg.load_from_ckpt = "best"
    eval_cfg.logger_name = []
    eval_cfg.valid_ds_params["n"] = 128
    task = run_task(0, eval_cfg)
    task.engines["evaluator"].run(task.dls["valid"], max_epochs=1)

    after = {p.name: p.stat().st_mtime for p in Path(cfg.save_dir).glob("*.pt")}
    assert after == before
    assert best.exists()
