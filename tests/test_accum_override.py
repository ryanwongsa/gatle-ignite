"""Accumulation lives in the base train_step, so overriding it would silently drop accum_steps."""

import warnings

import pytest
from examples.synthetic.configs.synthetic_v0 import get_config

from gatle_ignite.dispatch import import_entrypoint

BASE = get_config().main_runner


def _cfg(tmp_path, **overrides):
    cfg = get_config()
    cfg.save_dir = str(tmp_path)
    cfg.logger_name = []
    cfg.max_epochs = 1
    cfg.train_length = 2
    cfg.amp_dtype = "fp32"
    cfg.valid_ds_name = None
    cfg.every_val = 0
    cfg.score_name = None
    for key, value in overrides.items():
        cfg[key] = value
    return cfg


def _base():
    return import_entrypoint(BASE, "Trainer", field="main_runner")


def _overriding_trainer():
    class Overrides(_base()):
        def train_step(self, engine, batch, split="train"):
            return super().train_step(engine, batch, split=split)

    return Overrides


def _accum_warnings(trainer_cls, tmp_path, accum):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        trainer_cls(0, _cfg(tmp_path, accum_steps=accum)).setup()
    return [str(w.message) for w in caught if "accum_steps" in str(w.message)]


def test_overriding_train_step_with_accumulation_warns(tmp_path):
    hits = _accum_warnings(_overriding_trainer(), tmp_path, 4)
    assert hits, "accum_steps=4 silently did nothing and nobody said so"
    assert "train_step" in hits[0]


def test_no_warning_when_accumulation_is_off(tmp_path):
    assert not _accum_warnings(_overriding_trainer(), tmp_path, 1)


def test_no_warning_for_the_base_implementation(tmp_path):
    assert not _accum_warnings(_base(), tmp_path, 4)


def test_a_subclass_that_does_not_touch_train_step_does_not_warn(tmp_path):
    """Every task subclasses the trainer, so only overriding train_step may warn."""

    class JustPrepBatch(_base()):
        pass

    assert not _accum_warnings(JustPrepBatch, tmp_path, 4)


def test_a_bad_accum_steps_raises_rather_than_warning_about_train_step(tmp_path):
    """0, since ml_collections rejects a non-int on assignment, before this code runs."""
    from gatle_ignite import ConfigError

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with pytest.raises(ConfigError, match="accum_steps must be an integer >= 1"):
            _overriding_trainer()(0, _cfg(tmp_path, accum_steps=0)).setup()
    assert not [w for w in caught if "train_step" in str(w.message)]
