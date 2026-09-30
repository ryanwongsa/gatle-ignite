"""`init` writes a project that trains: the user's commands, run outside this repo."""

import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

# Everything except PYTHONPATH, so only the CLI's cwd-on-sys.path can resolve `models.mlp`.
# HOME stays: a user install lives under ~/.local, and the child still needs the framework.
_ENV = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}


def _run(*args, cwd):
    return subprocess.run(
        [sys.executable, "-m", "gatle_ignite", *args],
        cwd=cwd,
        capture_output=True,
        text=True,
        env=_ENV,
    )


@pytest.fixture(scope="module")
def project(tmp_path_factory):
    """A subprocess: in-process would skip _ensure_cwd_importable and leak sys.modules."""
    root = tmp_path_factory.mktemp("elsewhere")
    proc = _run("init", "demoproj", "--dir", str(root), cwd=root)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    return root / "demoproj"


def test_the_init_project_trains(project):
    proc = _run(
        "train", "--config=configs/demoproj_v0.py", "--override", "max_epochs=1", cwd=project
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "TRAINING COMPLETE" in proc.stdout


def test_the_config_preflights_without_crying_wolf(project):
    proc = _run("config", "--config=configs/demoproj_v0.py", cwd=project)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "all dotted paths resolve" in proc.stdout


def test_nothing_was_left_unsubstituted(project):
    """A placeholder in a .py fails the train test; one in the README is silent."""
    # The shared fixture's training run also left binary files here.
    written_by_init = [
        p
        for p in project.rglob("*")
        if p.is_file()
        and "checkpoints" not in p.parts
        and "__pycache__" not in p.parts
        and p.suffix in ("", ".py", ".md")
    ]
    assert written_by_init, "found nothing init wrote: the glob is wrong, not the template"

    leaked = [p for p in written_by_init if "{{" in p.read_text(encoding="utf-8")]
    assert not leaked, f"un-substituted placeholder in: {leaked}"
    survived = [p for p in project.rglob("*") if p.name.endswith("-tpl")]
    assert not survived, f"a template suffix reached the project: {survived}"


def test_the_project_ignores_its_own_checkpoints(project):
    ignored = (project / ".gitignore").read_text()
    assert "checkpoints/" in ignored, ".gitignore must cover the directory base_utils writes to"


def _shape(pkg):
    """The .py files under pkg, minus __init__.py, which only an example (a package) needs."""
    return {
        str(p.relative_to(pkg))
        for p in pkg.rglob("*.py")
        if "__pycache__" not in p.parts and p.name != "__init__.py"
    }


def test_the_template_is_the_synthetic_example(tmp_path):
    """Named `synthetic`, so both trees compare exactly with no renaming to get wrong."""
    proc = _run("init", "synthetic", "--dir", str(tmp_path), cwd=tmp_path)
    assert proc.returncode == 0, proc.stdout + proc.stderr

    written = _shape(tmp_path / "synthetic")
    example = _shape(ROOT / "examples" / "synthetic")
    assert written == example, (
        "the init template and examples/synthetic have drifted:\n"
        f"  only in the template: {sorted(written - example)}\n"
        f"  only in the example:  {sorted(example - written)}\n"
        "They must be the same shape: first-task.md documents the example as the shape "
        "init writes, so a difference here makes that page a lie."
    )


def test_init_refuses_a_name_that_cannot_be_imported(tmp_path):
    for name in ("9lives", "my-project", "class"):
        proc = _run("init", name, "--dir", str(tmp_path), cwd=tmp_path)
        assert proc.returncode != 0, f"init accepted {name!r}"
        assert "not a usable project name" in proc.stderr, proc.stderr
        assert not (tmp_path / name).exists(), f"init created {name!r} after rejecting it"


def test_init_refuses_to_shadow_the_framework(tmp_path):
    """`gatle_ignite.optimizers.adamw` would silently resolve into the user's project."""
    proc = _run("init", "gatle_ignite", "--dir", str(tmp_path), cwd=tmp_path)
    assert proc.returncode != 0
    assert "shadow the framework" in proc.stderr, proc.stderr


def test_init_refuses_a_name_already_importable(tmp_path):
    proc = _run("init", "json", "--dir", str(tmp_path), cwd=tmp_path)
    assert proc.returncode != 0
    assert "already importable" in proc.stderr, proc.stderr


def test_init_does_not_overwrite_without_force(tmp_path):
    assert _run("init", "demo2", "--dir", str(tmp_path), cwd=tmp_path).returncode == 0

    marker = tmp_path / "demo2" / "models" / "mlp.py"
    marker.write_text("# my real model\n")

    proc = _run("init", "demo2", "--dir", str(tmp_path), cwd=tmp_path)
    assert proc.returncode != 0, "init silently overwrote an existing project"
    assert "already exist" in proc.stderr, proc.stderr
    assert marker.read_text() == "# my real model\n", "refused, but wrote anyway"

    assert _run("init", "demo2", "--dir", str(tmp_path), "--force", cwd=tmp_path).returncode == 0
    assert marker.read_text() != "# my real model\n", "--force did not overwrite"
