"""BaseTrainer paths the examples use: grad clipping, warm starts, the tester, teardown."""

import importlib

import pytest
import torch
from examples.synthetic.configs.synthetic_v0 import get_config

from gatle_ignite.cli.launch import run_task
from gatle_ignite.trainer.checkpointing import load_checkpoints


@pytest.fixture
def cfg(tmp_path):
    cfg = get_config()
    cfg.save_dir = str(tmp_path)
    cfg.max_epochs = 2
    cfg.logger_name = []
    cfg.train_ds_params["n"] = 128
    cfg.valid_ds_params["n"] = 64
    return cfg


def test_grad_clip_norm_actually_clips(cfg):
    cfg.grad_clip_norm = 1e-8  # so small that every step must be clipped to it
    task = importlib.import_module(cfg.main_runner).Trainer(0, cfg)
    task.setup()

    seen = []
    original = task.backward

    def spy(loss, step=True):
        original(loss, step=step)
        total = sum(
            p.grad.detach().float().norm(2) ** 2
            for p in task.model.parameters()
            if p.grad is not None
        )
        seen.append(float(total**0.5))

    task.backward = spy
    task.fit()

    assert seen, "backward never ran"
    # Clipping happens before optimizer.step(), so these post-step norms are the clipped ones.
    assert max(seen) <= 1e-6, f"grad norm {max(seen):.2e} exceeds the 1e-8 clip"


def test_grad_clip_value_is_wired(cfg):
    cfg.grad_clip_value = 1e-8
    task = run_task(0, cfg)
    assert task.grad_clip_value == 1e-8


def test_model_checkpoint_dir_warm_starts_from_weights_only(cfg, tmp_path):
    """The documented way to extend a run without inheriting a stale LR schedule."""
    run_task(0, cfg)
    from gatle_ignite.utils.checkpoints import find_latest_checkpoint

    ckpt = find_latest_checkpoint(cfg.save_dir)
    trained = torch.load(ckpt, map_location="cpu", weights_only=False)["model"]

    fresh = get_config()
    fresh.save_dir = str(tmp_path / "warm")
    fresh.logger_name = []
    fresh.max_epochs = 1
    fresh.resume = False
    fresh.model_checkpoint_dir = str(ckpt)
    fresh.train_ds_params["n"] = 128
    fresh.valid_ds_params["n"] = 64

    task = importlib.import_module(fresh.main_runner).Trainer(0, fresh)
    task.setup()
    load_checkpoints(task)

    loaded = task.model.state_dict()
    for k, v in trained.items():
        assert torch.equal(loaded[k], v), f"{k} was not warm-started from the checkpoint"
    # Weights only: the epoch counter must NOT come along.
    assert task.engines["trainer"].state is None or task.engines["trainer"].state.epoch == 0


def test_model_checkpoint_dir_honours_strict_false(cfg, tmp_path, capsys):
    """Fine-tuning a new head: a checkpoint missing a tensor loads the rest, and says so."""
    source = importlib.import_module(cfg.main_runner).Trainer(0, cfg)
    source.setup()
    shifted = {k: v + 1 for k, v in source.model.state_dict().items()}
    dropped = sorted(shifted)[0]
    partial = {k: v for k, v in shifted.items() if k != dropped}
    path = tmp_path / "partial.pt"
    torch.save({"model": partial}, path)

    cfg.model_checkpoint_dir = str(path)
    cfg.strict = False
    task = importlib.import_module(cfg.main_runner).Trainer(0, cfg)
    task.setup()
    load_checkpoints(task)

    out = capsys.readouterr().out
    assert f"LOADED {len(partial)}/{len(shifted)} tensors" in out, out
    for key, value in partial.items():
        assert torch.equal(task.model.state_dict()[key], value), f"{key} did not load"


def test_tester_engine_runs_when_configured(cfg):
    cfg.test_ds_name = cfg.valid_ds_name
    cfg.test_ds_params = dict(cfg.valid_ds_params)
    cfg.every_test = 1
    cfg.tester_metrics = cfg.val_metrics
    task = run_task(0, cfg)
    assert task.engines["tester"] is not None
    assert "test/acc" in task.engines["tester"].state.metrics


def test_teardown_reports_failure_to_the_loggers(cfg):
    task = importlib.import_module(cfg.main_runner).Trainer(0, cfg)
    task.setup()

    reported = {}

    class Spy:
        def start(self):
            return self

        def on_train_epoch_end(self, trainer, optimizer):
            pass

        def on_train_iteration(self, trainer, model):
            pass

        def on_valid_epoch_end(self, trainer, engine):
            pass

        def on_completion(self, trainer):
            pass

        def finish(self, failed=False):
            reported["failed"] = failed

    task.logger = Spy()

    def boom(engine, batch):
        raise RuntimeError("simulated OOM at epoch 1")

    task.engines["trainer"]._process_function = boom

    with pytest.raises(RuntimeError, match="simulated OOM"):
        task.fit()
    assert reported.get("failed") is True, "the run died but was reported as a success"


def test_a_single_process_run_never_reaches_for_dataparallel(cfg, monkeypatch):
    """With several GPUs visible, auto_model picks DataParallel for a single process.

    DataParallel rejects auto_model_params' find_unused_parameters, and CI has no GPU to reach
    that branch, so this asserts auto_model is never called.
    """
    import gatle_ignite.trainer.base as base_mod

    def explode(*args, **kwargs):
        raise AssertionError(
            f"auto_model called for a single-process run with {kwargs} -- on a multi-GPU "
            f"host it would pick DataParallel and reject find_unused_parameters"
        )

    monkeypatch.setattr(base_mod.idist, "auto_model", explode)

    task = importlib.import_module(cfg.main_runner).Trainer(0, cfg)
    task.setup()

    assert not isinstance(task.model, torch.nn.DataParallel)
    assert not isinstance(task.model, torch.nn.parallel.DistributedDataParallel)
    # Device placement is the one thing auto_model did for a single process.
    assert all(p.device == task.device for p in task.model.parameters())


def test_the_shipped_default_is_what_would_have_broken_dataparallel(cfg):
    """If find_unused_parameters leaves the default, the test above proves nothing."""
    from gatle_ignite.config.base import base_config

    params = dict(base_config().auto_model_params)
    assert "find_unused_parameters" in params, (
        "the shipped default must carry find_unused_parameters, or "
        "test_a_single_process_run_never_reaches_for_dataparallel proves nothing"
    )
    assert "find_unused_parameters" not in str(torch.nn.DataParallel.__init__.__doc__ or ""), (
        "DataParallel grew a find_unused_parameters argument; the premise has changed"
    )
