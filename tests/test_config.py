import warnings

import pytest

from gatle_ignite import base_config, dispatch
from gatle_ignite.config.loader import apply_overrides, load_config_from_file
from gatle_ignite.config.validate import check_dispatch, validate_config
from gatle_ignite.dispatch import ConfigError


def _minimal():
    cfg = base_config()
    cfg.name = "t"
    cfg.save_dir = "/tmp/t"
    cfg.main_runner = "examples.synthetic.trainer.synthetic_trainer"
    cfg.model_name = "examples.synthetic.models.mlp"
    cfg.train_ds_name = "examples.synthetic.dataloaders.synthetic_dataset"
    cfg.criterion_name = "gatle_ignite.losses.composite"
    cfg.optimizer_name = "gatle_ignite.optimizers.adamw"
    return cfg


def test_minimal_config_validates():
    assert validate_config(_minimal()) is not None


def test_missing_required_fields_are_reported_together():
    cfg = base_config()
    cfg.name = "t"
    with pytest.raises(ConfigError) as e:
        validate_config(cfg)
    message = str(e.value)
    for field in ("save_dir", "main_runner", "model_name"):
        assert field in message


def test_an_unknown_top_level_field_is_ignored_silently():
    """A field the framework does not read is the project's, even one near a known name."""
    cfg = _minimal()
    cfg.max_epoch = 5
    cfg.my_data_root = "/data"
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        validate_config(cfg)


def test_overrides_are_literal_parsed():
    cfg = _minimal()
    apply_overrides(cfg, ["max_epochs=7", "logger_name=['text']"])
    assert cfg.max_epochs == 7
    assert cfg.logger_name == ["text"]


def test_override_of_unknown_field_raises():
    with pytest.raises(ConfigError, match="nonexistent"):
        apply_overrides(_minimal(), ["nonexistent=1"])


def test_config_check_resolves_every_dotted_path():
    cfg = _minimal()
    cfg.criterion_params = {
        "dict_of_loss_params": {
            "ce": {
                "cls_name": "examples.synthetic.losses.loss_functions.cross_entropy",
                "loss_params": {},
                "weight": 1.0,
            }
        }
    }
    cfg.val_metrics = {"acc": {"cls_name": "examples.synthetic.metrics.accuracy", "params": {}}}
    assert check_dispatch(cfg) is cfg


def test_config_check_reports_every_broken_path_at_once():
    cfg = _minimal()
    cfg.model_name = "examples.synthetic.modle"  # typo
    cfg.optimizer_name = "examples.synthetic.models.mlp"  # real module, no get_optimizer
    with pytest.raises(ConfigError) as e:
        check_dispatch(cfg)
    message = str(e.value)
    assert "2 dotted path(s)" in message
    assert "model_name" in message and "optimizer_name" in message


def test_config_check_reads_its_fields_from_the_contracts_table(monkeypatch):
    row = ("my_thing_name", "Thing", "A test-only contract.", ("my_thing_name",))
    monkeypatch.setattr(dispatch, "CONTRACTS", dispatch.CONTRACTS + (row,))
    cfg = _minimal()
    cfg.my_thing_name = "nope.not_a_module"
    with pytest.raises(ConfigError, match="my_thing_name"):
        check_dispatch(cfg)


def test_config_check_covers_the_nine_top_level_component_fields():
    fields = (
        "main_runner",
        "model_name",
        "train_ds_name",
        "valid_ds_name",
        "test_ds_name",
        "aug_name",
        "criterion_name",
        "optimizer_name",
        "lr_scheduler",
    )
    cfg = _minimal()
    for field in fields:
        cfg[field] = "nope.not_a_module"
    with pytest.raises(ConfigError) as e:
        check_dispatch(cfg)
    message = str(e.value)
    assert "9 dotted path(s)" in message
    for field in fields:
        assert f"config field '{field}'" in message


def test_config_check_leaves_collate_fn_to_dataloader_build():
    cfg = _minimal()
    cfg.train_ds_params = {"collate_fn": {"cls_name": "nope.not_a_module"}}
    assert check_dispatch(cfg) is cfg


def test_override_reaches_into_a_nested_field():
    """lr and batch size live in nested params dicts, so a sweep needs this."""
    cfg = _minimal()
    cfg.optimizer_params = {"lr": 1e-3, "weight_decay": 0.01}
    apply_overrides(cfg, ["optimizer_params.lr=1e-4"])
    assert cfg.optimizer_params.lr == 1e-4
    assert cfg.optimizer_params.weight_decay == 0.01  # siblings untouched


def test_override_of_an_unknown_nested_field_is_an_error():
    """A typo like `optimizer_params.lr_rate` must not add a key nothing reads."""
    cfg = _minimal()
    cfg.optimizer_params = {"lr": 1e-3}
    with pytest.raises(ConfigError, match="has no 'lr_rate'"):
        apply_overrides(cfg, ["optimizer_params.lr_rate=1e-4"])
    assert "lr_rate" not in cfg.optimizer_params


def test_override_through_a_non_mapping_says_so():
    cfg = _minimal()
    with pytest.raises(ConfigError, match="not a group of fields"):
        apply_overrides(cfg, ["max_epochs.foo=1"])


def test_a_config_using_a_relative_import_says_so(tmp_path):
    """The config is loaded by path, so it alone has no parent package to import from."""
    (tmp_path / "vocab.py").write_text("THING = 1\n")
    cfg = tmp_path / "cfg.py"
    cfg.write_text("from .vocab import THING\n\ndef get_config():\n    pass\n")

    with pytest.raises(ConfigError, match="relative import"):
        load_config_from_file(str(cfg))


def test_another_import_error_is_still_a_config_error(tmp_path):
    cfg = tmp_path / "cfg.py"
    cfg.write_text("import definitely_not_a_real_module\n\ndef get_config():\n    pass\n")

    with pytest.raises(ConfigError, match="could not import"):
        load_config_from_file(str(cfg))


@pytest.mark.parametrize("n_saved", [0, -1])
def test_a_non_positive_n_saved_is_rejected_naming_the_field(n_saved):
    """ignite >= 0.5.4 raises on this inside Checkpoint, after the run has started."""
    cfg = _minimal()
    cfg.n_saved = n_saved

    with pytest.raises(ConfigError, match="n_saved must be an int >= 1"):
        validate_config(cfg)


def test_n_saved_none_keeps_every_checkpoint_and_is_allowed():
    """None is ignite's own "keep them all" on every version in the supported range."""
    cfg = _minimal()
    cfg.n_saved = None

    validate_config(cfg)


def test_n_saved_is_not_policed_when_nothing_is_being_saved():
    """The message suggests save_ckpt = False, so it must not fire once that is set."""
    cfg = _minimal()
    cfg.save_ckpt = False
    cfg.n_saved = 0

    validate_config(cfg)


def test_a_non_integer_n_saved_is_a_config_error_not_a_traceback():
    cfg = _minimal()
    with cfg.ignore_type():
        cfg.n_saved = "two"

    with pytest.raises(ConfigError, match="n_saved must be an int >= 1"):
        validate_config(cfg)
