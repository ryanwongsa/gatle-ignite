"""Declaring an engine is one declaration, and the framework cannot silently forget it."""

import pytest
import torch
from examples.synthetic.configs.synthetic_v0 import get_config
from torch.utils.data import DataLoader, TensorDataset

from gatle_ignite import BaseTrainer, ConfigError, EngineSpec, to_device
from gatle_ignite.trainer.specs import EVAL_SPECS


class SpyLogger:
    """Records what a real LoggingCallback would have been asked to log."""

    def __init__(self):
        self.logged = {}

    def _record(self, trainer, engine):
        @engine.on(__import__("ignite.engine", fromlist=["Events"]).Events.EPOCH_COMPLETED)
        def _grab(e):
            self.logged.update(e.state.metrics)

    def start(self):
        return self

    def on_train_epoch_end(self, trainer, optimizer):
        pass

    def on_train_iteration(self, trainer, model):
        pass

    def on_valid_epoch_end(self, trainer, engine):
        self._record(trainer, engine)

    def on_completion(self, trainer):
        pass

    def finish(self, failed=False):
        pass


ACC = {
    "cls_name": "examples.synthetic.metrics.accuracy",
    "params": {"src_name": "logits", "tgt_name": ("targets", "labels")},
}


@pytest.fixture
def cfg(tmp_path):
    cfg = get_config()
    cfg.save_dir = str(tmp_path)
    cfg.max_epochs = 1
    cfg.logger_name = []
    cfg.train_ds_params["n"] = 128
    cfg.valid_ds_params["n"] = 64
    return cfg


class ExtraEngineTrainer(BaseTrainer):
    def prep_batch(self, batch, split="train", **kwargs):
        x, y = batch
        return to_device({"model_input": {"x": x}, "targets": {"labels": y}, "split": split})

    def eval_specs(self):
        # Shares the valid loader, so the test needs no extra dataset module.
        return super().eval_specs() + (EngineSpec.for_split("probe", ds_prefix="valid"),)


def _probe_cfg(cfg):
    cfg.probe_metrics = {"acc": ACC}
    cfg.every_probe = 1
    return cfg


def test_a_declared_engine_is_built_and_run(cfg):
    task = ExtraEngineTrainer(0, _probe_cfg(cfg))
    task.setup()
    assert task.engines["probe"] is not None
    task.fit()
    assert "probe/acc" in task.engines["probe"].state.metrics


def test_a_declared_engine_is_logged(cfg):
    spy = SpyLogger()
    task = ExtraEngineTrainer(0, _probe_cfg(cfg))
    task.setup()
    task.logger = spy
    task.fit()

    assert "probe/acc" in spy.logged, (
        "the declared engine ran but was never logged: the exact silent failure "
        f"this design exists to prevent. Logged: {sorted(spy.logged)}"
    )
    assert "valid/acc" in spy.logged  # and the built-ins still work


def test_a_declared_engine_can_score_its_own_checkpoint(cfg, tmp_path):
    cfg = _probe_cfg(cfg)
    cfg.probe_score_name = "probe/acc"
    cfg.probe_score_factor = 1
    task = ExtraEngineTrainer(0, cfg)
    task.fit()
    written = [p.name for p in tmp_path.iterdir()]
    assert any(n.startswith("probe_best_result") for n in written), written


def test_a_declared_engine_that_never_runs_warns_its_score_is_never_kept(cfg):
    """The scoring warnings cover the trainer's own engines, not only the built-ins."""
    cfg.probe_metrics = {"acc": ACC}
    cfg.probe_score_name = "probe/acc"
    cfg.every_probe = 0
    with pytest.warns(UserWarning, match="no best-on-probe checkpoint"):
        ExtraEngineTrainer(0, cfg)


def test_a_declared_engine_without_a_cadence_field_is_an_error(cfg):
    cfg.probe_metrics = {"acc": ACC}  # every_probe deliberately absent
    task = ExtraEngineTrainer(0, cfg)
    with pytest.raises(ConfigError, match="every_probe"):
        task.setup()


def test_two_engines_cannot_share_a_namespace(cfg):
    class T(ExtraEngineTrainer):
        def eval_specs(self):
            return EVAL_SPECS + (EngineSpec.for_split("valid", ds_prefix="valid"),)

    cfg.valid_metrics = {"acc": ACC}
    cfg.every_valid = 1
    task = T(0, cfg)
    with pytest.raises(ConfigError, match="engine_type"):
        task.setup()


def test_prep_batch_is_told_which_split_it_is_on(cfg):
    """The tester and the evaluator share eval_step, so only `split` tells them apart."""
    seen = []

    class T(BaseTrainer):
        def prep_batch(self, batch, split="train", **kwargs):
            seen.append(split)
            x, y = batch
            return to_device({"model_input": {"x": x}, "targets": {"labels": y}})

    cfg.test_ds_name = cfg.valid_ds_name
    cfg.test_ds_params = dict(cfg.valid_ds_params)
    cfg.every_test = 1
    cfg.tester_metrics = {"acc": ACC}
    T(0, cfg).fit()

    assert "train" in seen and "valid" in seen and "test" in seen, sorted(set(seen))


def test_each_engine_runs_its_own_dataloader(cfg):
    """_attach_eval_runner is a method so each handler gets its own scope, not the last spec."""
    ran = []

    class T(ExtraEngineTrainer):
        def build_dataloaders(self):
            dls, infos = super().build_dataloaders()
            # A probe loader that is distinguishable by length from the valid one.
            x = torch.randn(32, 64)
            y = torch.zeros(32, dtype=torch.long)
            self.dls["probe"] = DataLoader(TensorDataset(x, y), batch_size=8)
            self.infos["probe"] = {"length": 32}
            return dls, infos

        def eval_specs(self):
            return EVAL_SPECS + (EngineSpec.for_split("probe"),)

        def eval_step(self, engine, batch, split="valid"):
            ran.append((split, len(batch[0])))
            return super().eval_step(engine, batch, split=split)

    cfg = _probe_cfg(cfg)
    cfg.valid_ds_params["bs"] = 64  # 64 samples / bs 64 -> one batch of 64
    T(0, cfg).fit()

    assert ("probe", 8) in ran, f"probe engine did not run its own loader: {set(ran)}"
    assert ("valid", 64) in ran, f"valid engine did not run its own loader: {set(ran)}"


def warnings_as_errors():
    import warnings as _w

    class _Ctx:
        def __enter__(self):
            self._cm = _w.catch_warnings()
            self._cm.__enter__()
            _w.simplefilter("error")

        def __exit__(self, *a):
            return self._cm.__exit__(*a)

    return _Ctx()
