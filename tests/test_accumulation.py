"""Accumulating N micro-batches of B must land on the same weights as one batch of N*B."""

import copy

import pytest
import torch
from examples.synthetic.configs.synthetic_v0 import get_config

from gatle_ignite import ConfigError
from gatle_ignite.dispatch import import_entrypoint


def _cfg(tmp_path, *, bs, accum, n=64, epochs=1):
    cfg = get_config()
    cfg.save_dir = str(tmp_path)
    cfg.logger_name = []
    cfg.max_epochs = epochs
    cfg.amp_dtype = "fp32"
    cfg.seed = 7
    cfg.lr_scheduler = None  # a schedule would confound the LR
    cfg.optimizer_name = "gatle_ignite.optimizers.sgd"  # stateless: no momentum to diverge
    cfg.optimizer_params = {"lr": 0.1}
    cfg.accum_steps = accum
    cfg.train_ds_params = {
        **cfg.train_ds_params,
        "n": n,
        "bs": bs,
        "shuffle": False,
        "drop_last": False,
    }
    cfg.valid_ds_name = None
    cfg.every_val = 0
    cfg.score_name = None
    return cfg


def _train(cfg):
    Trainer = import_entrypoint(cfg.main_runner, "Trainer", field="main_runner")
    task = Trainer(0, cfg)
    task.setup()
    # cfg.seed already gives both runs the same init; assert it rather than trust it.
    initial = copy.deepcopy(task.model.state_dict())
    task.fit()
    return initial, task.model.state_dict()


@pytest.mark.parametrize("accum", [2, 4])
def test_accumulation_matches_the_big_batch_it_emulates(tmp_path, accum):
    big_init, big = _train(_cfg(tmp_path / "big", bs=64, accum=1))
    small_init, small = _train(_cfg(tmp_path / "small", bs=64 // accum, accum=accum))

    for key in big_init:
        assert torch.equal(big_init[key], small_init[key]), f"runs began differently: {key}"

    for key in big:
        assert torch.allclose(big[key], small[key], atol=1e-6), (
            f"{key} diverged: accumulating {accum} x {64 // accum} did not match one "
            f"batch of 64.\n  big:   {big[key].flatten()[:4]}\n  small: {small[key].flatten()[:4]}"
        )


def test_forgetting_to_divide_the_loss_would_be_caught(tmp_path):
    """Positive control: the equivalence test must notice an undivided (summed) loss."""

    class Bug(import_entrypoint(get_config().main_runner, "Trainer", field="main_runner")):
        def backward(self, loss, step=True):
            super().backward(loss * self.accum_steps, step=step)  # undo the division

    cfg = _cfg(tmp_path / "bug", bs=16, accum=4)
    task = Bug(0, cfg)
    task.setup()
    task.fit()
    bugged = task.model.state_dict()

    _, correct = _train(_cfg(tmp_path / "ok", bs=64, accum=1))
    assert not all(torch.allclose(correct[k], bugged[k], atol=1e-6) for k in correct), (
        "the equivalence test cannot see a 4x-too-large step, so it proves nothing"
    )


def test_a_short_final_window_still_steps(tmp_path):
    """10 batches at accum_steps=4 -> windows of 4, 4, 2. The 2 must not be discarded."""
    steps = []
    base = import_entrypoint(get_config().main_runner, "Trainer", field="main_runner")

    class Counter(base):
        def backward(self, loss, step=True):
            super().backward(loss, step=step)
            if step:
                steps.append(self.engines["trainer"].state.iteration)

    cfg = _cfg(tmp_path / "short", bs=8, accum=4, n=80)
    cfg.train_length = 10  # 10 batches per epoch: 4 + 4 + 2
    Counter(0, cfg).fit()

    assert steps == [4, 8, 10], (
        f"expected optimizer steps after batches 4, 8 and 10 (the short window), got {steps}"
    )


def test_the_window_is_sized_to_the_short_tail(tmp_path):
    """The short window must divide by ITS size, not by accum_steps."""
    base = import_entrypoint(get_config().main_runner, "Trainer", field="main_runner")
    windows = []

    class Spy(base):
        def _accum_window(self, engine):
            window, is_first, is_step = super()._accum_window(engine)
            windows.append(window)
            return window, is_first, is_step

    cfg = _cfg(tmp_path / "w", bs=8, accum=4, n=80)
    cfg.train_length = 10
    Spy(0, cfg).fit()

    assert windows == [4, 4, 4, 4, 4, 4, 4, 4, 2, 2], windows


def test_accum_steps_is_validated(tmp_path):
    cfg = _cfg(tmp_path / "bad", bs=8, accum=1)
    cfg.accum_steps = 0
    Trainer = import_entrypoint(cfg.main_runner, "Trainer", field="main_runner")
    with pytest.raises(ConfigError, match="accum_steps"):
        Trainer(0, cfg).setup()


def test_the_reported_loss_is_unscaled(tmp_path):
    """Runs that differ only in accum_steps must report comparable train/loss_avg."""
    _, _ = _train(_cfg(tmp_path / "a", bs=64, accum=1))
    Trainer = import_entrypoint(get_config().main_runner, "Trainer", field="main_runner")

    cfg = _cfg(tmp_path / "b", bs=16, accum=4)
    cfg.train_metrics = {}
    task = Trainer(0, cfg)
    task.fit()
    accum_loss = task.engines["trainer"].state.metrics["train/loss_avg"]

    cfg2 = _cfg(tmp_path / "c", bs=16, accum=1)
    task2 = Trainer(0, cfg2)
    task2.fit()
    plain_loss = task2.engines["trainer"].state.metrics["train/loss_avg"]

    # Same batches, same first epoch: the losses are comparable, not scaled by 1/4.
    assert accum_loss == pytest.approx(plain_loss, rel=0.5), (
        f"accum loss {accum_loss} vs plain {plain_loss}: looks scaled by accum_steps"
    )
