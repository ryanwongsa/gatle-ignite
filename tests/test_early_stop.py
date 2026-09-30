"""Early stopping, through a real `fit()`: a unit test of the counter passes unregistered."""

import importlib
import pathlib
import sys

import pytest
import torch
from examples.synthetic.configs.synthetic_v0 import get_config
from ignite.engine import Engine, State

from gatle_ignite.cli.launch import run_task
from gatle_ignite.trainer.checkpointing import _to_save
from gatle_ignite.utils.checkpoints import find_latest_checkpoint

FLAT = "valid/flat"

# Written into a throwaway project on sys.path: a `tests.` dotted path does not resolve
# everywhere the suite runs.
FLAT_METRIC = """
from ignite.metrics import Metric


class _Flat(Metric):
    def reset(self): pass
    def update(self, output): pass
    def compute(self): return 0.5


def get_metric(engine_type, info, **params):
    return _Flat(output_transform=lambda out: (out["y_pred"]["logits"], out["target"]["targets"]["labels"]))
"""

CLIMBING_METRIC = """
from ignite.metrics import Metric


class _Climb(Metric):
    calls = 0

    def reset(self): pass
    def update(self, output): pass

    def compute(self):
        type(self).calls += 1
        return float(type(self).calls)


def get_metric(engine_type, info, **params):
    return _Climb(output_transform=lambda out: (out["y_pred"]["logits"], out["target"]["targets"]["labels"]))
"""


@pytest.fixture
def metric_module(tmp_path, monkeypatch):

    def _write(name, source):
        (tmp_path / f"{name}.py").write_text(source)
        monkeypatch.syspath_prepend(str(tmp_path))
        sys.modules.pop(name, None)
        return name

    return _write


@pytest.fixture
def cfg(tmp_path, metric_module):
    cfg = get_config()
    cfg.save_dir = str(tmp_path)
    cfg.max_epochs = 20
    cfg.logger_name = []
    cfg.train_ds_params["n"] = 128
    cfg.valid_ds_params["n"] = 64
    cfg.val_metrics = {
        "flat": {"cls_name": metric_module("flat_metric", FLAT_METRIC), "params": {}}
    }
    cfg.score_name = FLAT
    cfg.score_factor = 1
    return cfg


def _epochs(task):
    return task.engines["trainer"].state.epoch


def test_a_flat_metric_stops_the_run(cfg):
    cfg.early_stop_patience = 3
    task = run_task(0, cfg)
    # Epoch 1 sets the best and epochs 2 to 4 fail to beat it.
    assert _epochs(task) == 4, "patience did not end the run where the count says it should"
    assert _epochs(task) < cfg.max_epochs


def test_the_default_trains_the_whole_schedule(cfg):
    task = run_task(0, cfg)
    assert cfg.early_stop_patience == 0
    assert _epochs(task) == cfg.max_epochs, "a config setting neither field must be unaffected"


def test_the_floor_blocks_a_stop_patience_alone_would_have_taken(cfg):
    """`early_stop_after` gives a cosine tail its chance to move."""
    cfg.early_stop_patience = 3
    cfg.early_stop_after = 12
    task = run_task(0, cfg)
    assert _epochs(task) > 4, "the floor did not delay the counter"
    # Epoch 12 sets the best and epochs 13 to 15 fail to beat it.
    assert _epochs(task) == 15


def test_patience_counts_evaluations_not_epochs(cfg):
    cfg.every_val = 3
    cfg.early_stop_patience = 2
    task = run_task(0, cfg)
    assert _epochs(task) == 9, "patience counted epochs rather than this engine's runs"


def test_the_counter_rides_in_the_checkpoint(cfg):
    cfg.early_stop_patience = 8
    cfg.max_epochs = 4
    run_task(0, cfg)
    state = torch.load(find_latest_checkpoint(cfg.save_dir), map_location="cpu")
    assert "early_stop_valid" in state, "the counter is not in the checkpoint at all"
    # Epoch 1 sets the best and epochs 2 to 4 fail to beat it.
    assert state["early_stop_valid"]["counter"] == 3
    assert state["early_stop_valid"]["best_score"] == 0.5


def test_a_resumed_run_continues_its_count_rather_than_restarting(cfg, tmp_path):
    """A long run that resumes often must still accumulate enough patience."""
    cfg.early_stop_patience = 5
    cfg.max_epochs = 4
    run_task(0, cfg)

    resumed = get_config()
    resumed.save_dir = str(tmp_path)
    resumed.logger_name = []
    resumed.train_ds_params["n"] = 128
    resumed.valid_ds_params["n"] = 64
    resumed.val_metrics = dict(cfg.val_metrics)
    resumed.score_name = FLAT
    resumed.early_stop_patience = 5
    resumed.max_epochs = 20
    whole = resumed.copy_and_resolve_references()
    whole.save_dir = str(tmp_path / "uninterrupted")
    resumed.resume = True
    task = run_task(0, resumed)
    uninterrupted = run_task(0, whole)
    # Epoch 1 sets the best and epochs 2 to 6 are the five failures.
    assert _epochs(uninterrupted) == 6
    assert _epochs(task) == _epochs(uninterrupted), "the resumed run lost or gained a count"


def test_the_tester_cannot_stop_a_run():
    """Letting a test split end training would be selecting on it."""
    from gatle_ignite.enhancements.early_stop import early_stop_settings
    from gatle_ignite.trainer.specs import EVAL_SPECS

    tester = EVAL_SPECS[1]
    assert tester.engine_type == "test"
    assert tester.early_stop_patience_field is None
    assert early_stop_settings(tester, get_config()) is None


def test_a_declared_engine_gets_the_fields_from_one_declaration():
    from gatle_ignite.trainer.specs import EngineSpec

    spec = EngineSpec.for_split("probe")
    assert spec.early_stop_patience_field == "probe_early_stop_patience"
    assert spec.early_stop_after_field == "probe_early_stop_after"


def test_it_says_it_stopped_and_where(cfg, capsys):
    cfg.early_stop_patience = 3
    run_task(0, cfg)
    out = capsys.readouterr().out
    assert "[early-stop]" in out
    assert "stopping at epoch 4" in out, "the run did not say where it stopped"


def test_a_score_that_keeps_improving_is_never_stopped(cfg, metric_module):
    cfg.val_metrics = {
        "climb": {"cls_name": metric_module("climbing_metric", CLIMBING_METRIC), "params": {}}
    }
    cfg.score_name = "valid/climb"
    cfg.early_stop_patience = 2
    task = run_task(0, cfg)
    assert _epochs(task) == cfg.max_epochs, "an improving run was stopped"


def test_the_stopping_epoch_still_writes_its_checkpoints(cfg):
    """The filenames are the evidence: the live engine's epoch is right even when none is saved."""
    cfg.early_stop_patience = 3
    task = run_task(0, cfg)
    assert task.engines["trainer"].state.epoch == 4

    written = sorted(pathlib.Path(cfg.save_dir).glob("*.pt"))
    names = [f.name for f in written]
    assert any(n.startswith("latest_epoch_checkpoint_4") for n in names), names
    assert any(n.startswith("valid_best_result_checkpoint_4") for n in names), names
    state = torch.load(find_latest_checkpoint(cfg.save_dir), map_location="cpu")
    assert "model" in state and "optimizer" in state


def test_one_engine_stopping_does_not_make_another_claim_it_did(cfg, capsys, metric_module):
    """`trainer.should_terminate` is set by whoever terminated, not by this handler."""
    import importlib

    from gatle_ignite.trainer.specs import EVAL_SPECS, EngineSpec

    flat = metric_module("flat_metric", FLAT_METRIC)
    cfg.early_stop_patience = 3
    cfg.probe_metrics = {"flat": {"cls_name": flat, "params": {}}}
    cfg.every_probe = 1
    cfg.probe_score_name = "probe/flat"
    cfg.probe_score_factor = 1
    cfg.probe_early_stop_patience = 9
    cfg.probe_early_stop_after = 0
    cfg.probe_ds_name = cfg.valid_ds_name
    cfg.probe_ds_params = dict(cfg.valid_ds_params)

    base = importlib.import_module(cfg.main_runner).Trainer

    class TwoStoppers(base):
        def eval_specs(self):
            return EVAL_SPECS + (EngineSpec.for_split("probe"),)

    task = TwoStoppers(0, cfg)
    task.fit()

    out = capsys.readouterr().out
    assert "stopping at epoch 4" in out, out
    assert "probe: no improvement" not in out, (
        "the probe engine claimed to have stopped a run its counter never reached"
    )
    assert "probe: 3/9" in out, "the probe's own honest count was swallowed"


def test_a_patience_with_no_score_to_watch_is_refused(cfg):
    from gatle_ignite.dispatch import ConfigError

    cfg.score_name = None
    cfg.early_stop_patience = 3
    with pytest.raises(ConfigError, match="no metric to stop on"):
        run_task(0, cfg)


def test_early_stopping_adds_no_checkpoint_key_when_unconfigured(cfg):
    cfg.max_epochs = 2
    run_task(0, cfg)
    state = torch.load(find_latest_checkpoint(cfg.save_dir), map_location="cpu")
    assert not [k for k in state if k.startswith("early_stop")], (
        "an unconfigured run gained a checkpoint key, so every older checkpoint would "
        "warn about missing state on resume"
    )


def test_it_is_inert_under_evaluate(cfg):
    """evaluate() never calls attach_runner."""
    cfg.early_stop_patience = 1
    cfg.max_epochs = 2
    run_task(0, cfg)

    cfg.run = True
    task = importlib.import_module(cfg.main_runner).Trainer(0, cfg)
    task.evaluate()
    assert task._post_eval == {}, "an inference run registered a stopper"


def test_a_retry_after_the_config_error_still_gets_its_stopper(cfg):
    """A caller that catches the error, fixes the config and rebuilds must get a stopper."""
    import importlib

    from gatle_ignite.dispatch import ConfigError

    cfg.early_stop_patience = 3
    cfg.score_name = None
    task = importlib.import_module(cfg.main_runner).Trainer(0, cfg)
    with pytest.raises(ConfigError):
        task.setup()

    # setup() is guarded by _built, which the raise left False, so it really re-runs.
    task.cfg.score_name = FLAT
    task.setup()
    assert list(task._early_stoppers) == ["early_stop_valid"]
    assert "early_stop_valid" in _to_save(task)


def test_stopping_on_an_engine_that_never_runs_is_refused(cfg):
    """every_val = 0 builds no evaluator."""
    from gatle_ignite.dispatch import ConfigError

    cfg.every_val = 0
    cfg.early_stop_patience = 3
    with pytest.raises(ConfigError, match="never builds"):
        run_task(0, cfg)


def test_the_score_is_broadcast_even_when_rank_zero_raises(cfg, monkeypatch):
    """Otherwise every other rank blocks in broadcast until the NCCL timeout."""
    import ignite.distributed as idist

    from gatle_ignite.enhancements.early_stop import EarlyStop

    sent = []
    monkeypatch.setattr(idist, "get_world_size", lambda: 2)
    monkeypatch.setattr(idist, "get_rank", lambda: 0)
    monkeypatch.setattr(idist, "broadcast", lambda v, src=0: sent.append(v) or v)

    def boom(engine):
        raise RuntimeError("score_name names a metric this engine never produced")

    trainer = Engine(lambda e, b: None)
    trainer.state = State(epoch=1)
    stopper = EarlyStop(spec=object(), patience=2, after=0, score_function=boom, trainer=trainer)
    with pytest.raises(RuntimeError, match="never produced"):
        stopper._agreed_score(object())
    assert sent == [0.0], "rank 0 raised without broadcasting; the other ranks would hang"


def test_only_rank_zero_announces(cfg, monkeypatch, capsys):
    import ignite.distributed as idist

    from gatle_ignite.enhancements.early_stop import EarlyStop

    monkeypatch.setattr(idist, "get_world_size", lambda: 2)
    monkeypatch.setattr(idist, "get_rank", lambda: 1)
    monkeypatch.setattr(idist, "broadcast", lambda v, src=0: 0.5)

    trainer = Engine(lambda e, b: None)
    trainer.state = State(epoch=9)
    stopper = EarlyStop(
        spec=object(), patience=1, after=0, score_function=lambda e: 0.5, trainer=trainer
    )
    stopper(object())
    stopper(object())
    assert capsys.readouterr().out == "", "a non-zero rank announced the verdict"


def test_a_checkpoint_from_an_older_ignite_still_resumes(cfg):
    """ignite 0.5.5 added `threshold_mode`, so older checkpoints need padding to load.

    Stripping to the two keys ignite <= 0.5.4 wrote reproduces that writer on any version.
    """
    cfg.early_stop_patience = 8
    cfg.max_epochs = 4
    run_task(0, cfg)
    state = torch.load(find_latest_checkpoint(cfg.save_dir), map_location="cpu")
    written = dict(state["early_stop_valid"])
    older = {k: written[k] for k in ("counter", "best_score")}

    fresh = get_config()
    fresh.save_dir = cfg.save_dir
    fresh.logger_name = []
    fresh.early_stop_patience = 8
    fresh.score_name = cfg.score_name
    fresh.score_factor = cfg.score_factor
    task = importlib.import_module(fresh.main_runner).Trainer(0, fresh)
    task.setup()
    stopper = task._early_stoppers["early_stop_valid"]

    stopper.load_state_dict(older)

    assert stopper.state_dict()["counter"] == written["counter"]
    assert stopper.state_dict()["best_score"] == written["best_score"]


def test_a_checkpoint_missing_the_counter_is_rejected_rather_than_reset(cfg):
    """Padding fills only keys ignite added later: a lost `counter` must fail, not reset to 0."""
    cfg.early_stop_patience = 8
    cfg.max_epochs = 4
    run_task(0, cfg)
    state = torch.load(find_latest_checkpoint(cfg.save_dir), map_location="cpu")
    damaged = {k: v for k, v in state["early_stop_valid"].items() if k != "counter"}

    fresh = get_config()
    fresh.save_dir = cfg.save_dir
    fresh.logger_name = []
    fresh.early_stop_patience = 8
    fresh.score_name = cfg.score_name
    fresh.score_factor = cfg.score_factor
    task = importlib.import_module(fresh.main_runner).Trainer(0, fresh)
    task.setup()
    stopper = task._early_stoppers["early_stop_valid"]

    with pytest.raises(ValueError, match="counter"):
        stopper.load_state_dict(damaged)
