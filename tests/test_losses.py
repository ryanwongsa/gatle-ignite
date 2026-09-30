import pytest
import torch

from gatle_ignite import get_value
from gatle_ignite.dispatch import ConfigError
from gatle_ignite.losses.composite import Loss as Composite

CE = "examples.synthetic.losses.loss_functions.cross_entropy"


def _entry(weight=1.0, **extra):
    return {
        "cls_name": CE,
        "loss_params": {"src_name": "logits", "tgt_name": ("targets", "labels")},
        "weight": weight,
        **extra,
    }


def _batch():
    y_pred = {"logits": torch.randn(8, 4)}
    target = {"targets": {"labels": torch.randint(0, 4, (8,))}}
    return y_pred, target


def test_get_value_resolves_string_and_tuple_paths():
    container = {"a": 1, "b": {"c": {"d": 2}}}
    assert get_value(container, "a") == 1
    assert get_value(container, ("b", "c", "d")) == 2


def test_get_value_reports_where_a_path_broke():
    with pytest.raises(ConfigError, match="b -> nope"):
        get_value({"b": {}}, ("b", "nope", "d"))


def test_weight_scales_the_logged_component():
    y_pred, target = _batch()
    torch.manual_seed(0)
    full = Composite({"ce": _entry(weight=1.0)})
    half = Composite({"ce": _entry(weight=0.5)})
    _, d_full = full(y_pred, target)
    total_half, d_half = half(y_pred, target)
    assert d_half["loss_ce"] == pytest.approx(d_full["loss_ce"].item() * 0.5, rel=1e-5)
    # the total is the sum of the (already weighted) components
    assert total_half.item() == pytest.approx(d_half["loss_ce"].item(), rel=1e-5)


def test_total_is_the_sum_of_components():
    y_pred, target = _batch()
    loss = Composite({"a": _entry(weight=1.0), "b": _entry(weight=2.0)})
    total, components = loss(y_pred, target)
    assert total.item() == pytest.approx(sum(v.item() for v in components.values()), rel=1e-5)


def test_crit_keys_drives_metric_naming():
    loss = Composite({"ce": _entry(), "aux": _entry()})
    assert loss.crit_keys == ["ce", "aux"]


def test_iteration_gating_disables_a_term_until_its_start():
    y_pred, target = _batch()
    loss = Composite({"ce": _entry(), "late": _entry(start_iteration=100)})

    _, early = loss(y_pred, target, iteration=1)
    assert early["loss_late"].item() == 0.0

    _, later = loss(y_pred, target, iteration=101)
    assert later["loss_late"].item() > 0.0


def test_submodules_are_registered_so_they_move_with_the_module():
    """A plain dict of sub-losses would leave their buffers behind on .to(device)."""
    loss = Composite({"ce": _entry()})
    assert any("dict_loss_functions.ce" in name for name, _ in loss.named_modules())


def test_empty_composite_is_rejected():
    with pytest.raises(ConfigError, match="at least one sub-loss"):
        Composite({})


def test_entry_missing_weight_is_rejected():
    with pytest.raises(ConfigError, match="weight"):
        Composite({"ce": {"cls_name": CE, "loss_params": {}}})
