"""A config can choose the collate_fn, and a task can choose which train losses it reports."""

from dataclasses import replace

import pytest
import torch
from examples.synthetic.configs.synthetic_v0 import get_config
from torch.utils.data import TensorDataset

from gatle_ignite import ConfigError
from gatle_ignite.data.helpers import build_dataloader, resolve_collate_fn
from gatle_ignite.dispatch import import_entrypoint

# ---- collate_fn ---------------------------------------------------


def _ds(n=8):
    return TensorDataset(torch.arange(n).float().unsqueeze(1), torch.zeros(n).long())


def test_a_plain_callable_still_works():
    marker = {}

    def my_collate(batch):
        marker["called"] = True
        return torch.utils.data._utils.collate.default_collate(batch)

    dl = build_dataloader(_ds(), {"bs": 4, "collate_fn": my_collate})
    next(iter(dl))
    assert marker.get("called"), "a callable collate_fn was not used"


def test_a_config_can_name_a_collate_by_dotted_path():
    dl = build_dataloader(
        _ds(),
        {"bs": 4, "collate_fn": {"cls_name": "tests.collate_probe", "params": {"tag": 7}}},
    )
    batch = next(iter(dl))
    assert batch["tag"] == 7, "the dotted-path collate did not reach the DataLoader"


def test_the_dotted_form_takes_no_params():
    dl = build_dataloader(_ds(), {"bs": 4, "collate_fn": {"cls_name": "tests.collate_probe"}})
    assert next(iter(dl))["tag"] == 0  # the entrypoint's own default


def test_no_collate_fn_means_torchs_own():
    assert resolve_collate_fn({"bs": 4}) is None


def test_a_broken_collate_path_names_the_field():
    with pytest.raises(ConfigError, match="collate_fn.cls_name"):
        resolve_collate_fn({"collate_fn": {"cls_name": "nope.not.a.module"}})


def test_a_collate_that_is_neither_says_so():
    """Neither a callable nor a dispatch dict: name both shapes, not fail later in torch."""
    with pytest.raises(ConfigError, match="must be a callable or"):
        resolve_collate_fn({"collate_fn": "myproj.data.pad_collate"})  # a bare string


def test_the_exact_sharded_eval_path_honours_it_too():
    """The eval path builds its own DataLoader around ExactDistributedSampler."""
    params = {"bs": 4, "exact_sharding": True, "collate_fn": {"cls_name": "tests.collate_probe"}}
    dl = build_dataloader(_ds(), params)  # world_size == 1 here, so the normal path
    assert next(iter(dl))["tag"] == 0

    from gatle_ignite.data.helpers import _build_exact_sharded_loader

    assert next(iter(_build_exact_sharded_loader(_ds(), params)))["tag"] == 0


# ---- with_losses --------------------------------------------------


def _cfg(tmp_path):
    cfg = get_config()
    cfg.save_dir = str(tmp_path)
    cfg.logger_name = []
    cfg.max_epochs = 1
    cfg.train_length = 2
    cfg.amp_dtype = "fp32"
    cfg.valid_ds_name = None
    cfg.every_val = 0
    cfg.score_name = None
    return cfg


def _base():
    return import_entrypoint(get_config().main_runner, "Trainer", field="main_runner")


def test_the_default_still_reports_the_loss(tmp_path):
    task = _base()(0, _cfg(tmp_path))
    task.fit()
    assert "train/loss_avg" in task.engines["trainer"].state.metrics


def test_a_task_can_drop_the_total_and_keep_the_components(tmp_path):
    """A GAN has no total loss worth printing, but loss_d and loss_g are."""

    class NoTotal(_base()):
        def train_spec(self):
            return replace(super().train_spec(), total_loss=False)

    task = NoTotal(0, _cfg(tmp_path))
    task.fit()
    metrics = task.engines["trainer"].state.metrics
    assert "train/loss_avg" not in metrics, (
        f"total_loss=False still reported the total: {sorted(metrics)}"
    )
    # The synthetic criterion declares crit_keys=["ce"].
    assert "train/loss_ce_avg" in metrics, (
        f"total_loss=False also dropped the components: {sorted(metrics)}"
    )


def test_with_losses_false_drops_everything(tmp_path):
    """The eval specs use this: their step computes no loss at all."""

    class NoLosses(_base()):
        def train_spec(self):
            return replace(super().train_spec(), with_losses=False)

    task = NoLosses(0, _cfg(tmp_path))
    task.fit()
    metrics = task.engines["trainer"].state.metrics
    assert not [k for k in metrics if "loss" in k], sorted(metrics)


def test_turning_it_off_does_not_break_training(tmp_path):
    """The loss still drives the optimizer; only the report goes away."""

    class NoTotal(_base()):
        def train_spec(self):
            return replace(super().train_spec(), total_loss=False)

    task = NoTotal(0, _cfg(tmp_path))
    task.setup()
    before = {k: v.clone() for k, v in task.model.state_dict().items()}
    task.fit()
    assert any(not torch.equal(before[k], v) for k, v in task.model.state_dict().items()), (
        "no weight moved: total_loss=False disabled training, not just its report"
    )


def test_the_train_spec_hook_is_what_builds_the_engine(tmp_path):
    """The engine itself comes from the hook, not just its metrics."""
    seen = []

    class Spy(_base()):
        def train_spec(self):
            spec = super().train_spec()
            seen.append(spec.key)
            return spec

    Spy(0, _cfg(tmp_path)).setup()
    assert "trainer" in seen
