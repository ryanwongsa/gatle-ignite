import pytest
from examples.synthetic.configs.synthetic_v0 import get_config

from gatle_ignite import BaseTrainer
from gatle_ignite.dispatch import ConfigError

ACC = "examples.synthetic.metrics.accuracy"


@pytest.fixture
def get_metrics():
    trainer = BaseTrainer(0, get_config())

    def build(spec, engine_type):
        return trainer.dict_metric_from_list(engine_type, spec, {})

    return build


def test_metrics_are_namespaced_by_engine_so_score_name_resolves(get_metrics):
    metrics = get_metrics({"acc": {"cls_name": ACC, "params": {}}}, "valid")
    assert "valid/acc" in metrics


def test_metric_spec_accepts_a_configdict(get_metrics):
    """ml_collections turns an assigned dict into a ConfigDict, not a dict subclass."""
    from ml_collections import config_dict

    spec = config_dict.ConfigDict({"acc": {"cls_name": ACC, "params": {}}})
    assert "test/acc" in get_metrics(spec, "test")


def test_bare_list_of_names_is_rejected_with_the_escape_hatch_named(get_metrics):
    with pytest.raises(ConfigError, match="dict_metric_from_list"):
        get_metrics(["acc"], "valid")


def test_metric_entry_without_cls_name_is_rejected(get_metrics):
    with pytest.raises(ConfigError, match="cls_name"):
        get_metrics({"acc": {"params": {}}}, "valid")


def test_empty_spec_is_fine(get_metrics):
    assert get_metrics({}, "train") == {}
