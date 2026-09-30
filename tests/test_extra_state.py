"""extra_to_save() state rides in the checkpoint and comes back on save, resume and eval."""

import pytest
import torch
from examples.synthetic.configs.synthetic_v0 import get_config

from gatle_ignite import ConfigError
from gatle_ignite.dispatch import import_entrypoint
from gatle_ignite.trainer.checkpointing import load_checkpoints, load_eval_checkpoint

BASE = get_config().main_runner


class Counter:
    """The smallest thing ignite's Checkpoint accepts: state_dict/load_state_dict."""

    def __init__(self, value=0):
        self.value = value

    def state_dict(self):
        return {"value": self.value}

    def load_state_dict(self, state):
        self.value = state["value"]


def _trainer_class(**extra):
    base = import_entrypoint(BASE, "Trainer", field="main_runner")

    class WithExtra(base):
        def __init__(self, local_rank, cfg):
            super().__init__(local_rank, cfg)
            self.counter = Counter()

        def extra_to_save(self):
            return {**extra, "counter": self.counter}

        def backward(self, loss, step=True):
            super().backward(loss, step=step)
            if step:
                self.counter.value += 1

    return WithExtra


def _cfg(tmp_path, **overrides):
    cfg = get_config()
    cfg.save_dir = str(tmp_path)
    cfg.logger_name = []
    cfg.max_epochs = 1
    cfg.train_length = 3
    cfg.val_length = 1
    cfg.amp_dtype = "fp32"
    for key, value in overrides.items():
        cfg[key] = value
    return cfg


def test_extra_state_reaches_the_checkpoint(tmp_path):
    task = _trainer_class()(0, _cfg(tmp_path))
    task.fit()
    assert task.counter.value == 3  # 3 batches, 3 optimizer steps

    saved = torch.load(next(tmp_path.glob("latest_epoch*.pt")), map_location="cpu")
    assert "counter" in saved, f"extra state missing from the checkpoint: {sorted(saved)}"
    assert saved["counter"] == {"value": 3}


def test_extra_state_comes_back_on_resume(tmp_path):
    _trainer_class()(0, _cfg(tmp_path)).fit()

    fresh = _trainer_class()(0, _cfg(tmp_path, resume=True, max_epochs=1))
    assert fresh.counter.value == 0  # proves the assert below is not vacuous
    fresh.setup()
    load_checkpoints(fresh)
    assert fresh.counter.value == 3, "resume rebuilt the counter instead of restoring it"


def test_extra_state_comes_back_for_eval(tmp_path):
    _trainer_class()(0, _cfg(tmp_path)).fit()

    fresh = _trainer_class()(0, _cfg(tmp_path, run=True, load_from_ckpt="latest"))
    fresh.setup()
    assert fresh.counter.value == 0
    load_eval_checkpoint(fresh)
    assert fresh.counter.value == 3, (
        "gatle-ignite eval rebuilt the extra state instead of loading it: an EMA run "
        "would score its raw weights and label them valid/acc"
    )


def test_an_old_checkpoint_without_the_extra_state_still_evaluates(tmp_path):
    plain = import_entrypoint(BASE, "Trainer", field="main_runner")(0, _cfg(tmp_path))
    plain.fit()  # written before extra_to_save() existed, so it has no "counter"

    fresh = _trainer_class()(0, _cfg(tmp_path, run=True, load_from_ckpt="latest"))
    fresh.setup()
    with pytest.warns(UserWarning, match="has no 'counter'"):
        load_eval_checkpoint(fresh)


def test_colliding_with_the_frameworks_own_state_is_an_error(tmp_path):
    """Silently winning costs the framework's state; silently losing costs yours."""
    task = _trainer_class(model=Counter())(0, _cfg(tmp_path))
    with pytest.raises(ConfigError, match="collides"):
        task.fit()


def test_colliding_with_conditional_state_is_caught_too(tmp_path):
    """ "scheduler" is added only when a config schedules, so check after adding it."""
    task = _trainer_class(scheduler=Counter())(0, _cfg(tmp_path))
    assert task.cfg.lr_scheduler is not None, "test needs a scheduler"
    with pytest.raises(ConfigError, match="collides"):
        task.fit()
