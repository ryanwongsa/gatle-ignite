import pytest

from gatle_ignite.dispatch import ConfigError, import_entrypoint, import_module_from_path


def test_imports_a_real_entrypoint():
    Model = import_entrypoint("examples.synthetic.models.mlp", "Model")
    assert Model(in_dim=4, n_classes=2) is not None


def test_missing_module_names_the_config_field():
    with pytest.raises(ConfigError, match="model_name"):
        import_module_from_path("nope.not.here", field="model_name")


def test_missing_attribute_reports_the_contract_and_what_was_found():
    with pytest.raises(ConfigError) as e:
        import_entrypoint("examples.synthetic.models.mlp", "get_optimizer", field="optimizer_name")
    assert "must expose 'get_optimizer'" in str(e.value)
    assert "Model" in str(e.value)  # lists what the module does export


@pytest.mark.parametrize("bad", [None, 42, ""])
def test_non_string_paths_are_rejected(bad):
    with pytest.raises(ConfigError):
        import_module_from_path(bad)
