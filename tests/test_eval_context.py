"""Weights swapped in for evaluation (an EMA shadow, say) must never reach the checkpoint.

ignite fires EPOCH_COMPLETED inside run(), so the checkpoint is written after run() returns,
outside eval_context, whatever anyone registers.
"""

import contextlib
import glob

import pytest
import torch
from examples.synthetic.configs.synthetic_v0 import get_config

from gatle_ignite import EngineSpec
from gatle_ignite.dispatch import import_entrypoint

BASE = get_config().main_runner


def _cfg(tmp_path, **overrides):
    cfg = get_config()
    cfg.save_dir = str(tmp_path)
    cfg.logger_name = []
    cfg.max_epochs = 1
    cfg.train_length = 2
    cfg.val_length = 2
    cfg.every_val = 1
    cfg.amp_dtype = "fp32"
    cfg.lr_scheduler = None
    for key, value in overrides.items():
        cfg[key] = value
    return cfg


def _base():
    return import_entrypoint(BASE, "Trainer", field="main_runner")


def _zeroing_trainer():
    """A trainer whose eval_context zeroes every weight: a swap you cannot miss."""

    class Zeroing(_base()):
        @contextlib.contextmanager
        def eval_context(self, spec):
            live = {k: v.clone() for k, v in self.model.state_dict().items()}
            self.model.load_state_dict({k: torch.zeros_like(v) for k, v in live.items()})
            try:
                yield
            finally:
                self.model.load_state_dict(live)

    return Zeroing


def test_the_best_checkpoint_stores_live_weights_not_the_swapped_ones(tmp_path):
    task = _zeroing_trainer()(0, _cfg(tmp_path, score_name="valid/acc"))
    task.fit()

    best = glob.glob(str(tmp_path / "valid_best_result*.pt"))
    assert best, "no best checkpoint was written"
    saved = torch.load(best[0], map_location="cpu")["model"]

    assert not all(bool(saved[k].eq(0).all()) for k in saved), (
        "the best checkpoint stored the SWAPPED (zeroed) model; the live weights are gone"
    )
    live = task.model.state_dict()
    for key in saved:
        assert torch.equal(saved[key], live[key].cpu()), f"{key} is not the live weight"


def test_metrics_are_measured_inside_the_eval_context(tmp_path):
    """A zeroed model cannot beat chance, so this shows the context was open during batches."""
    task = _zeroing_trainer()(0, _cfg(tmp_path, score_name="valid/acc", val_length=4))
    task.fit()
    zeroed_acc = task.engines["evaluator"].state.metrics["valid/acc"]

    plain = _base()(0, _cfg(tmp_path / "plain", score_name="valid/acc", val_length=4))
    plain.fit()
    live_acc = plain.engines["evaluator"].state.metrics["valid/acc"]

    assert zeroed_acc != live_acc, (
        f"the swapped model scored the same as the live one ({zeroed_acc}): the "
        f"eval_context did not cover the iterations"
    )


def test_evaluate_enters_the_eval_context(tmp_path):
    """evaluate() never calls attach_runner, so both paths must go through run_eval_engine."""
    seen = []

    class Spy(_base()):
        @contextlib.contextmanager
        def eval_context(self, spec):
            seen.append(spec.key)
            yield

    _base()(0, _cfg(tmp_path, score_name="valid/acc")).fit()  # produce a checkpoint

    cfg = _cfg(tmp_path, run=True, load_from_ckpt="latest", test_ds_name=None)
    Spy(0, cfg).evaluate()
    assert "evaluator" in seen, f"evaluate() never entered eval_context (saw {seen})"


def test_an_exception_during_eval_releases_the_swap_and_propagates(tmp_path):
    """ignite re-raises only if no EXCEPTION_RAISED handler is registered on the engine."""
    state = {"swapped": False}

    class Boom(_base()):
        @contextlib.contextmanager
        def eval_context(self, spec):
            state["swapped"] = True
            try:
                yield
            finally:
                state["swapped"] = False

        def eval_step(self, engine, batch, split="valid"):
            raise RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom"):
        Boom(0, _cfg(tmp_path)).fit()

    assert state["swapped"] is False, "the swap leaked: the next epoch trains swapped weights"


def test_a_failed_eval_does_not_select_a_checkpoint(tmp_path):
    """A crashed eval has only partial metrics to score a checkpoint on."""

    class Boom(_base()):
        def eval_step(self, engine, batch, split="valid"):
            raise RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom"):
        Boom(0, _cfg(tmp_path, score_name="valid/acc")).fit()

    assert not glob.glob(str(tmp_path / "valid_best_result*.pt"))


def test_the_context_can_swap_for_one_engine_and_not_another(tmp_path):
    """The EMA example's `raw` engine reads the same data with the live weights."""
    raw = EngineSpec.for_split("raw", ds_prefix="valid")

    class Controlled(_zeroing_trainer()):
        def eval_specs(self):
            return super().eval_specs() + (raw,)

        @contextlib.contextmanager
        def eval_context(self, spec):
            if spec.key == "raw":
                yield  # the control: live weights
                return
            with super().eval_context(spec):
                yield

    cfg = _cfg(tmp_path, val_length=4)
    cfg.every_raw = 1
    cfg.raw_metrics = dict(cfg.val_metrics)
    task = Controlled(0, cfg)
    task.fit()

    assert (
        task.engines["evaluator"].state.metrics["valid/acc"]
        != (task.engines["raw"].state.metrics["raw/acc"])
    ), "the control engine saw the swapped model too: eval_context ignored spec"
