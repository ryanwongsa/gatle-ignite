"""Multi-node plumbing only: the topology maths and what reaches idist.Parallel.

Nothing here proves two nodes rendezvous; docs/status.md lists that as unverified.
"""

import os

import pytest
from examples.synthetic.configs.synthetic_v0 import get_config

from gatle_ignite import ConfigError
from gatle_ignite.cli import launch as launch_mod


@pytest.fixture
def cfg(tmp_path):
    cfg = get_config()
    cfg.save_dir = str(tmp_path)
    cfg.logger_name = []
    cfg.max_epochs = 1
    return cfg


class SpyParallel:
    """Stands in for idist.Parallel and records how it was constructed."""

    last = None

    def __init__(self, **kwargs):
        SpyParallel.last = kwargs

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def run(self, fn, cfg):
        SpyParallel.last["ran"] = True


# ---- topology ----------------------------------------------------------------


def test_defaults_are_one_process_on_one_node(cfg, monkeypatch):
    monkeypatch.setattr(launch_mod.torch.cuda, "device_count", lambda: 0)
    assert launch_mod.resolve_topology(cfg) == (1, 1, 0)


def test_one_process_per_visible_gpu_by_default(cfg, monkeypatch):
    monkeypatch.setattr(launch_mod.torch.cuda, "device_count", lambda: 4)
    assert launch_mod.resolve_topology(cfg) == (4, 1, 0)


def test_nproc_per_node_overrides_the_gpu_count(cfg, monkeypatch):
    monkeypatch.setattr(launch_mod.torch.cuda, "device_count", lambda: 4)
    cfg.nproc_per_node = 2
    assert launch_mod.resolve_topology(cfg) == (2, 1, 0)


@pytest.mark.parametrize(
    "nnodes,node_rank,match",
    [(0, 0, "nnodes must be"), (2, 2, "node_rank must be"), (2, -1, "node_rank must be")],
)
def test_bad_topology_is_rejected(cfg, nnodes, node_rank, match):
    cfg.nnodes = nnodes
    cfg.node_rank = node_rank
    with pytest.raises(ConfigError, match=match):
        launch_mod.resolve_topology(cfg)


# ---- rendezvous --------------------------------------------------------------


def test_multi_node_refuses_to_rendezvous_with_itself(cfg, monkeypatch):
    """Otherwise every node silently forms its own world and trains its own model."""
    monkeypatch.delenv("MASTER_ADDR", raising=False)
    cfg.nnodes = 2
    with pytest.raises(ConfigError, match="unrelated models"):
        launch_mod.resolve_rendezvous(cfg, nnodes=2)


def test_multi_node_refuses_an_unset_port(cfg, monkeypatch):
    """Otherwise each node picks its own random port and the rendezvous silently hangs."""
    monkeypatch.delenv("MASTER_PORT", raising=False)
    cfg.master_addr = "10.0.0.1"  # valid, so the addr guard passes and the port guard is reached
    with pytest.raises(ConfigError, match="master_port"):
        launch_mod.resolve_rendezvous(cfg, nnodes=2)


def test_single_node_is_happy_on_localhost(cfg, monkeypatch):
    monkeypatch.delenv("MASTER_ADDR", raising=False)
    addr, port = launch_mod.resolve_rendezvous(cfg, nnodes=1)
    assert addr == "127.0.0.1"
    assert 1 <= port <= 65535


def test_config_beats_the_environment(cfg, monkeypatch):
    monkeypatch.setenv("MASTER_ADDR", "10.0.0.9")
    monkeypatch.setenv("MASTER_PORT", "1234")
    cfg.master_addr = "10.0.0.1"
    cfg.master_port = 29500
    assert launch_mod.resolve_rendezvous(cfg, nnodes=2) == ("10.0.0.1", 29500)


def test_the_environment_is_used_when_the_config_is_silent(cfg, monkeypatch):
    monkeypatch.setenv("MASTER_ADDR", "10.0.0.9")
    monkeypatch.setenv("MASTER_PORT", "1234")
    assert launch_mod.resolve_rendezvous(cfg, nnodes=2) == ("10.0.0.9", 1234)


# ---- what actually reaches idist.Parallel ------------------------------------


def test_multi_node_passes_the_topology_through(cfg, monkeypatch):
    monkeypatch.setattr(launch_mod.torch.cuda, "device_count", lambda: 2)
    monkeypatch.setattr(launch_mod.idist, "Parallel", SpyParallel)
    cfg.nnodes = 3
    cfg.node_rank = 1
    cfg.master_addr = "10.0.0.1"
    cfg.master_port = 29500
    cfg.dist_backend = "gloo"

    launch_mod.launch(cfg)

    got = SpyParallel.last
    assert got["nnodes"] == 3
    assert got["node_rank"] == 1
    assert got["master_addr"] == "10.0.0.1"
    assert got["master_port"] == 29500
    assert got["nproc_per_node"] == 2
    assert got["backend"] == "gloo"
    # init_method would double-specify a rendezvous idist derives from the above.
    assert "init_method" not in got
    assert got["ran"] is True


def test_single_node_multi_gpu_names_the_rendezvous_without_init_method(cfg, monkeypatch):
    monkeypatch.setattr(launch_mod.torch.cuda, "device_count", lambda: 2)
    monkeypatch.setattr(launch_mod.idist, "Parallel", SpyParallel)
    monkeypatch.delenv("MASTER_ADDR", raising=False)

    launch_mod.launch(cfg)

    got = SpyParallel.last
    assert got["nproc_per_node"] == 2
    assert "nnodes" not in got
    assert got["master_addr"] == "127.0.0.1"
    assert got["master_port"] == int(os.environ["MASTER_PORT"])
    # ignite >= 0.5.5 rejects a tcp:// init_method outright, and every version before it
    # refuses one alongside master_port.
    assert "init_method" not in got


def test_a_single_process_run_never_touches_parallel(cfg, monkeypatch):
    """A laptop run must stay a plain function call: no rendezvous, no spawn."""
    monkeypatch.setattr(launch_mod.torch.cuda, "device_count", lambda: 0)

    def explode(**kwargs):
        raise AssertionError("single-process run must not construct idist.Parallel")

    monkeypatch.setattr(launch_mod.idist, "Parallel", explode)
    called = []
    monkeypatch.setattr(launch_mod, "run_task", lambda rank, c: called.append(rank))

    launch_mod.launch(cfg)
    assert called == [0]


# --- the one that actually spawns --------------------------------------------------
#
# Everything above only inspects SpyParallel's dict; this one reaches ignite's real spawn.


def _rendezvous_probe(local_rank, results):
    import os

    import ignite.distributed as idist

    results[idist.get_rank()] = (idist.get_world_size(), os.environ.get("MASTER_PORT"))


@pytest.mark.slow
def test_single_node_kwargs_are_accepted_by_a_real_spawn():
    """The port, not the world size: ignite <= 0.5.3 silently hardcodes 2222 otherwise."""
    import ignite.distributed as idist
    import torch.multiprocessing as mp

    kwargs = launch_mod._parallel_kwargs(nnodes=1, node_rank=0, addr="127.0.0.1", port=29513)
    assert "init_method" not in kwargs

    results = mp.Manager().dict()
    with idist.Parallel(backend="gloo", nproc_per_node=2, **kwargs) as parallel:
        parallel.run(_rendezvous_probe, results)

    assert dict(results) == {0: (2, "29513"), 1: (2, "29513")}
