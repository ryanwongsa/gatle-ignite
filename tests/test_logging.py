import csv
import subprocess
import sys
import textwrap

import pytest

from gatle_ignite.callbacks.logging import LoggingCallback, _flatten


def test_flatten_unwraps_tensors_and_nested_dicts():
    import torch

    flat = _flatten(
        {
            "valid/acc": torch.tensor(0.5),
            "valid/per_class": {"a": 1.0, "b": 2.0},
            "train/loss_avg": 0.25,
            "valid/summary_blob": "skipped",  # 'summary' keys are not scalar series
        }
    )
    assert flat["valid/acc"] == 0.5
    assert flat["valid/per_class_a"] == 1.0
    assert flat["train/loss_avg"] == 0.25
    assert "valid/summary_blob" not in flat


def test_unknown_backend_names_the_field(tmp_path):
    from gatle_ignite import base_config
    from gatle_ignite.dispatch import ConfigError

    cfg = base_config()
    cfg.name, cfg.save_dir = "t", str(tmp_path)
    cfg.logger_name = ["nope_not_a_backend"]
    with pytest.raises(ConfigError, match="logger_name"):
        LoggingCallback(cfg).start()


def test_a_custom_backend_is_dispatched_and_receives_its_metrics(tmp_path):
    """The `Backend` template, which test_templates.py cannot wire into its one config."""
    from gatle_ignite import base_config

    cfg = base_config()
    cfg.name, cfg.save_dir = "custom", str(tmp_path)
    cfg.logger_name = ["pbar", "examples.templates.backend"]

    cb = LoggingCallback(cfg).start()

    # "pbar" is a trainer flag, not a sink: one backend, not two.
    assert len(cb.backends) == 1
    assert type(cb.backends[0]).__module__ == "examples.templates.backend"

    cb._log({"train/loss": 0.5, "train/lr_0": 0.001}, step=1, epoch=1)
    cb.finish()

    csv_path = tmp_path / "custom_metrics.csv"
    assert csv_path.exists(), "the custom backend was constructed but never logged to"
    rows = list(csv.DictReader(csv_path.open()))
    assert len(rows) == 1
    assert rows[0]["train/loss"] == "0.5"
    assert rows[0]["epoch"] == "1"


@pytest.mark.slow
def test_optional_backends_are_not_imported_when_unused(tmp_path):
    """A subprocess, so another test's imports cannot mask a top-level `import wandb`."""
    script = textwrap.dedent(
        f"""
        import sys
        from examples.synthetic.configs.synthetic_v0 import get_config
        from gatle_ignite.cli.launch import run_task

        cfg = get_config()
        cfg.save_dir = {str(tmp_path)!r}
        cfg.max_epochs = 1
        cfg.logger_name = ["text"]
        cfg.train_ds_params["n"] = 128
        cfg.valid_ds_params["n"] = 64
        run_task(0, cfg)

        assert "wandb" not in sys.modules, "wandb was imported for a text-only run"
        assert "requests" not in sys.modules, "requests was imported for a text-only run"
        print("OK")
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, timeout=300
    )
    assert result.returncode == 0, result.stderr
    assert "OK" in result.stdout
