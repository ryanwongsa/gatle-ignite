"""`--config=a.py,b.py` runs both, each with its own MASTER_PORT.

Subprocesses, since the port allocation and the environment scrubbing happen in the child.
"""

import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

# Everything except PYTHONPATH, so the child resolves `examples.` via the CLI's cwd.
_ENV = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}

_LAUNCH_RE = re.compile(r"Launching: (?P<path>.+?) \(MASTER_PORT=(?P<port>\d+)\)")


def _run(*args, cwd=ROOT):
    return subprocess.run(
        [sys.executable, "-m", "gatle_ignite", *args],
        cwd=cwd,
        capture_output=True,
        text=True,
        env=_ENV,
    )


def _write_config(path, save_dir):
    """save_dir is in the file: every child gets the same `--override`s."""
    path.write_text(
        "from examples.synthetic.configs.synthetic_v0 import get_config as _base\n"
        "\n"
        "def get_config():\n"
        "    cfg = _base()\n"
        f"    cfg.name = {path.stem!r}\n"
        f"    cfg.save_dir = {str(save_dir)!r}\n"
        "    cfg.logger_name = []\n"
        "    return cfg\n"
    )
    return path


@pytest.mark.slow
def test_two_configs_run_in_parallel_each_with_its_own_port(tmp_path):
    a_dir, b_dir = tmp_path / "a_out", tmp_path / "b_out"
    a = _write_config(tmp_path / "cfg_a.py", a_dir)
    b = _write_config(tmp_path / "cfg_b.py", b_dir)

    proc = _run(
        "train",
        f"--config={a},{b}",
        "--override",
        "max_epochs=1",
        "--override",
        "train_length=2",
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr

    # A checkpoint is the only artifact a run that failed to start cannot produce.
    for out in (a_dir, b_dir):
        assert list(out.glob("latest_epoch_checkpoint_*.pt")), (
            f"no checkpoint in {out}: that config did not train\n{proc.stdout}"
        )

    # Two runs sharing a port collide during rendezvous.
    launched = _LAUNCH_RE.findall(proc.stdout)
    assert len(launched) == 2, f"expected 2 launch lines, got {launched}\n{proc.stdout}"
    ports = {int(port) for _path, port in launched}
    assert len(ports) == 2, f"both children were given the same MASTER_PORT: {ports}"
    assert all(49152 <= p <= 65535 for p in ports), f"ports outside the ephemeral range: {ports}"

    assert "[cfg_a]" in proc.stdout and "[cfg_b]" in proc.stdout


@pytest.mark.slow
def test_a_failing_config_is_named_and_the_exit_code_survives(tmp_path):
    ok_dir = tmp_path / "ok_out"
    ok = _write_config(tmp_path / "cfg_ok.py", ok_dir)
    bad = tmp_path / "cfg_bad.py"
    bad.write_text("import a_module_that_does_not_exist\n\ndef get_config():\n    pass\n")

    proc = _run(
        "train",
        f"--config={ok},{bad}",
        "--override",
        "max_epochs=1",
        "--override",
        "train_length=2",
    )

    assert proc.returncode == 1, f"a failing config exited {proc.returncode}\n{proc.stdout}"
    assert "Failed configs: cfg_bad" in proc.stdout, proc.stdout
    assert "cfg_ok" not in proc.stdout.split("Failed configs:")[-1], (
        "the passing config was reported as failed"
    )


def test_inherited_torchrun_rendezvous_vars_are_dropped(monkeypatch):
    """A child inheriting RANK & co. from torchrun/SLURM joins the parent's world."""
    from gatle_ignite.cli import launch as launch_mod

    captured = []

    class FakeProc:
        stdout = []

        def wait(self):
            return 0

    def fake_popen(cmd, **kwargs):
        captured.append(kwargs["env"])
        return FakeProc()

    monkeypatch.setattr(launch_mod.subprocess, "Popen", fake_popen)
    for var in ("RANK", "WORLD_SIZE", "LOCAL_RANK", "LOCAL_WORLD_SIZE", "GROUP_RANK", "ROLE_RANK"):
        monkeypatch.setenv(var, "7")

    assert launch_mod.launch_many(["a.py", "b.py"]) == 0
    assert len(captured) == 2

    for env in captured:
        for var in (
            "RANK",
            "WORLD_SIZE",
            "LOCAL_RANK",
            "LOCAL_WORLD_SIZE",
            "GROUP_RANK",
            "ROLE_RANK",
        ):
            assert var not in env, f"{var} leaked into the child: it would join the parent's world"
        assert env["MASTER_ADDR"] == "127.0.0.1"

    assert len({env["MASTER_PORT"] for env in captured}) == 2, "children shared a MASTER_PORT"


def test_the_ports_are_distinct_but_not_proven_free(monkeypatch):
    """pick_port() never binds, so a clash with another process stays possible."""
    from gatle_ignite.cli import launch as launch_mod

    captured = []

    class FakeProc:
        stdout = []

        def wait(self):
            return 0

    monkeypatch.setattr(
        launch_mod.subprocess,
        "Popen",
        lambda cmd, **kw: (captured.append(kw["env"]), FakeProc())[1],
    )

    # 40 draws from ~16k ports collide with probability ~5% if `used` were not checked.
    paths = [f"cfg_{i}.py" for i in range(40)]
    assert launch_mod.launch_many(paths) == 0
    ports = [env["MASTER_PORT"] for env in captured]
    assert len(set(ports)) == len(paths), "pick_port() handed the same port to two children"
