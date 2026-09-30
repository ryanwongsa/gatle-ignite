"""Every example config resolves. CI trains them; this catches a rename first, in seconds."""

import importlib
from pathlib import Path

import pytest

from gatle_ignite.config.validate import check_dispatch

ROOT = Path(__file__).resolve().parents[1]

# mnist is deliberately absent: it downloads, so it cannot run in a sandboxed test.
EXAMPLES = (
    "synthetic",
    "classification",
    "contrastive",
    "diffusion",
    "ema",
    "gan",
    "teacher_student",
    "translation",
)


def _cfg(name):
    return importlib.import_module(f"examples.{name}.configs.{name}_v0").get_config()


def _every_config_module():
    """Every config in the tree, such as teacher_student's teacher_v0, not just EXAMPLES."""
    for path in sorted((ROOT / "examples").glob("*/configs/*.py")):
        if path.stem in ("__init__", "base_utils"):  # not configs: no get_config()
            continue
        if path.parents[1].name == "mnist":  # see EXAMPLES
            continue
        yield f"examples.{path.parents[1].name}.configs.{path.stem}"


@pytest.mark.parametrize("dotted", sorted(_every_config_module()))
def test_every_config_in_the_tree_resolves(dotted):
    check_dispatch(importlib.import_module(dotted).get_config())


@pytest.mark.parametrize("dotted", sorted(_every_config_module()))
def test_no_example_would_commit_its_checkpoints(dotted):
    """Checks the outcome, not a call to ckpt_dir(): any path git ignores is fine."""
    import shutil
    import subprocess

    if shutil.which("git") is None:  # pragma: no cover
        pytest.skip("git is what defines the answer here")

    save_dir = str(importlib.import_module(dotted).get_config().save_dir)
    ignored = subprocess.run(["git", "check-ignore", "-q", save_dir], cwd=ROOT).returncode == 0
    assert ignored, (
        f"{dotted} writes checkpoints to {save_dir}, which .gitignore does not cover: "
        f"a training run would leave .pt files staged. Route save_dir through the "
        f"example's configs/base_utils.py ckpt_dir()."
    )


@pytest.mark.parametrize("name", EXAMPLES)
def test_every_example_config_resolves(name):
    """check_dispatch is what `gatle-ignite config` runs as a user's pre-flight."""
    check_dispatch(_cfg(name))
