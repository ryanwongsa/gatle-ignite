"""`cfg.compile`: the guard, checkpoints across the compile boundary, and the wiring.

Forcing compile ON patches `gatle_ignite.enhancements.compile.resolve_compile`, which only bites
while its callers look it up in `compile.py`'s own globals.
"""

import importlib

import pytest
import torch
import torch.nn as nn

from gatle_ignite import ConfigError
from gatle_ignite.enhancements.compile import _would_be_skipped, compile_model, is_compiled


class Plain(nn.Module):
    """A model torch.compile can reach: the forward is ours, not torch's."""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(8, 8), nn.ReLU(), nn.Linear(8, 4))

    def forward(self, x):
        return self.net(x)


# ----------------------------------------------------------------------------------
# The guard
# ----------------------------------------------------------------------------------


def test_a_model_compile_cannot_reach_is_refused():
    """torch.compile silently compiles nothing for a bare nn.Sequential."""
    with pytest.raises(ConfigError) as excinfo:
        compile_model(nn.Sequential(nn.Linear(4, 4)))
    message = str(excinfo.value)
    assert "cfg.compile" in message
    assert "Sequential" in message
    assert "nn.Module subclass" in message


def test_a_model_compile_can_reach_is_compiled_in_place():
    model = Plain()
    returned = compile_model(model)

    assert returned is model, "in place: the caller's object must be what got compiled"
    assert is_compiled(model)
    assert type(model) is Plain, "torch.compile(model) would have returned OptimizedModule"


def test_the_guard_would_reject_ddp_which_is_why_it_runs_before_wrapping():
    """A fact about torch, not this package: why build_model compiles before auto_model.

    Dynamo skips DDP's own forward, so the guard would refuse every wrapped model even though
    compiling one works. The gloo test below is the one that fails if the order changes.
    """
    from torch._dynamo.trace_rules import check
    from torch.nn.parallel import DistributedDataParallel

    assert check(DistributedDataParallel.forward) is True
    assert _would_be_skipped(nn.Sequential(nn.Linear(4, 4))) is True
    assert _would_be_skipped(Plain()) is False


def test_is_compiled_does_not_accept_a_compiled_submodule():
    """`self.encoder.compile()` leaves the top-level forward eager."""

    class Inner(nn.Module):
        def forward(self, x):
            return x

    class Outer(nn.Module):
        def __init__(self):
            super().__init__()
            self.enc = Inner()

        def forward(self, x):
            return self.enc(x)

    submodule_only = Outer()
    submodule_only.enc.compile()
    assert is_compiled(submodule_only) is False

    top_level = compile_model(Outer())
    assert is_compiled(top_level) is True


def test_is_compiled_sees_through_a_wrapper_but_not_the_torch_compile_one():
    """build_model compiles, then auto_model wraps, so that must answer True.

    The torch.compile wrapper answers False on purpose: ignite never strips its `_orig_mod.`
    prefix from a checkpoint.
    """

    class Wrapper(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, *args, **kwargs):
            return self.module(*args, **kwargs)

    assert is_compiled(Wrapper(compile_model(Plain()))) is True
    assert is_compiled(torch.compile(Plain())) is False


def test_a_missing_trace_rules_skips_the_guard_rather_than_crashing(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def refuse(name, *args, **kwargs):
        if name == "torch._dynamo.trace_rules":
            raise ImportError("pretend it moved")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", refuse)
    assert _would_be_skipped(nn.Sequential(nn.Linear(4, 4))) is False


# ----------------------------------------------------------------------------------
# Checkpoints
# ----------------------------------------------------------------------------------


def test_compiling_changes_no_state_dict_key():
    """Why compile is in place: ignite's Checkpoint knows nothing of OptimizedModule."""
    torch.manual_seed(0)
    eager = Plain()
    torch.manual_seed(0)
    compiled = compile_model(Plain())

    assert list(eager.state_dict()) == list(compiled.state_dict())
    assert not any(k.startswith("_orig_mod.") for k in compiled.state_dict())
    for key, tensor in eager.state_dict().items():
        assert torch.allclose(tensor, compiled.state_dict()[key])


@pytest.mark.parametrize("direction", ["compiled_to_eager", "eager_to_compiled"])
def test_a_checkpoint_round_trips_between_compiled_and_eager(direction):
    """Through Checkpoint.load_objects, which never calls _align_module_prefix."""
    from ignite.handlers import Checkpoint

    torch.manual_seed(0)
    source = Plain()
    torch.manual_seed(1)
    target = Plain()
    if direction == "compiled_to_eager":
        compile_model(source)
    else:
        compile_model(target)

    # Not already equal, or the load below would prove nothing.
    assert not torch.allclose(source.net[0].weight, target.net[0].weight)

    state = {"model": source.state_dict()}
    Checkpoint.load_objects(to_load={"model": target}, checkpoint=state)

    assert len(state["model"]) == len(target.state_dict())
    for key, tensor in source.state_dict().items():
        assert torch.allclose(tensor, target.state_dict()[key])


# ----------------------------------------------------------------------------------
# Wiring
# ----------------------------------------------------------------------------------


def _synthetic_cfg(tmp_path):
    from examples.synthetic.configs.synthetic_v0 import get_config

    cfg = get_config()
    cfg.save_dir = str(tmp_path)
    cfg.max_epochs = 1
    cfg.train_length = 1
    cfg.val_length = 1
    cfg.logger_name = []
    # Above the batch size, or drop_last leaves every split empty.
    cfg.train_ds_params["n"] = 256
    cfg.valid_ds_params["n"] = 256
    cfg.compile = True
    return cfg


def test_the_default_build_model_compiles(tmp_path):
    cfg = _synthetic_cfg(tmp_path)
    task = importlib.import_module(cfg.main_runner).Trainer(0, cfg)
    task.setup()
    assert is_compiled(task.model)


def test_a_build_model_override_that_ignores_the_flag_is_refused(tmp_path):
    """An override that drops the flag differs from one that honours it only in step time."""
    cfg = _synthetic_cfg(tmp_path)
    base = importlib.import_module(cfg.main_runner).Trainer

    class Uncompiled(base):
        def build_model(self):
            from gatle_ignite.dispatch import import_entrypoint

            Model = import_entrypoint(self.cfg.model_name, "Model", field="model_name")
            return Model(**self.cfg.model_params)

    with pytest.raises(ConfigError, match="does not consider compiled") as excinfo:
        Uncompiled(0, cfg).setup()
    assert "torch.compile(model)" in str(excinfo.value)
    assert "self.encoder.compile()" in str(excinfo.value)


def test_compile_off_leaves_the_model_untouched(tmp_path):
    cfg = _synthetic_cfg(tmp_path)
    cfg.compile = False
    task = importlib.import_module(cfg.main_runner).Trainer(0, cfg)
    task.setup()
    assert not is_compiled(task.model)


@pytest.mark.parametrize("trained_compiled", [True, False])
@pytest.mark.parametrize("resumed_compiled", [True, False])
def test_resume_crosses_the_compile_boundary_in_both_directions(
    tmp_path, trained_compiled, resumed_compiled
):
    """`_resume_from` filters keys and restores the epoch, beyond what load_objects does.

    All four combinations, since a prefix added on save but not on load breaks one way only.
    """
    from gatle_ignite.cli.launch import run_task

    first = _synthetic_cfg(tmp_path)
    first.compile = trained_compiled
    first.max_epochs = 1
    first.train_length = None
    run_task(0, first)

    second = _synthetic_cfg(tmp_path)
    second.compile = resumed_compiled
    second.max_epochs = 2
    second.train_length = None
    second.resume = True
    task = run_task(0, second)

    assert task.engines["trainer"].state.epoch == 2, "resume restarted from scratch"
    assert is_compiled(task.model) is resumed_compiled


@pytest.mark.parametrize("trained_compiled", [True, False])
@pytest.mark.parametrize("eval_compiled", [True, False])
def test_eval_ckpt_best_crosses_the_compile_boundary(tmp_path, trained_compiled, eval_compiled):
    """`load_eval_checkpoint` is a third load path, separate from resume and load_weights."""
    from gatle_ignite.cli.launch import run_task

    trained = _synthetic_cfg(tmp_path)
    trained.compile = trained_compiled
    trained.max_epochs = 1
    trained.train_length = None
    run_task(0, trained)

    evaluated = _synthetic_cfg(tmp_path)
    evaluated.compile = eval_compiled
    evaluated.run = True
    evaluated.load_from_ckpt = "best"
    task = run_task(0, evaluated)

    assert task.engines["trainer"].state.epoch == 0, "eval mode must not train"
    assert task.dls["valid"] is not None


@pytest.mark.slow
def test_compile_trains_under_two_rank_gloo(tmp_path):
    """Fails if build_model runs the guard after auto_model; no single process can see it."""
    import ignite.distributed as idist
    import torch.multiprocessing as mp

    manager = mp.Manager()
    results = manager.dict()
    with idist.Parallel(backend="gloo", nproc_per_node=2) as parallel:
        parallel.run(_gloo_worker, str(tmp_path), results)

    collected = dict(results)
    assert set(collected) == {0, 1}
    assert all(r["world_size"] == 2 for r in collected.values())
    assert all(r["compiled"] for r in collected.values())
    # ignite strips `module.` on save; `_orig_mod.` would mean the torch.compile wrapper.
    assert all(not r["orig_mod_keys"] for r in collected.values())


def _gloo_worker(local_rank, save_dir, results):
    import ignite.distributed as idist

    from gatle_ignite.cli.launch import run_task

    cfg = _synthetic_cfg(save_dir)
    cfg.auto_model_params = {"find_unused_parameters": False, "sync_bn": False}
    task = run_task(local_rank, cfg)
    results[idist.get_rank()] = {
        "world_size": idist.get_world_size(),
        "compiled": is_compiled(task.model),
        "orig_mod_keys": [k for k in task.model.state_dict() if "_orig_mod." in k],
    }


# ----------------------------------------------------------------------------------
# cfg.compile = "auto"
# ----------------------------------------------------------------------------------


def test_auto_compile_is_off_without_an_accelerator(tmp_path, capsys):
    """ "auto" answers "can this run compile", never "should it", so it never times anything."""
    from gatle_ignite.enhancements.compile import resolve_compile

    cfg = _synthetic_cfg(tmp_path)
    cfg.compile = "auto"
    assert resolve_compile(cfg) is False
    assert "auto -> OFF" in capsys.readouterr().out, "a silent auto decision is the failure mode"


def test_auto_compile_turns_itself_off_for_a_model_dynamo_would_skip(tmp_path, monkeypatch, capsys):
    """The model lives in a temp dir: a `tests` package on sys.path may not be this repo's."""
    import ignite.distributed as idist

    (tmp_path / "skipped_model.py").write_text(
        "import torch.nn as nn\n\n\n"
        "def Model(**kwargs):\n"
        "    # A bare container: dynamo skips torch's own forward, so compiling it is a\n"
        "    # no-op. cfg.compile = True refuses this; 'auto' resolves around it.\n"
        "    return nn.Sequential(nn.Linear(64, 128), nn.ReLU(), nn.Linear(128, 10))\n"
    )
    monkeypatch.syspath_prepend(str(tmp_path))

    cfg = _synthetic_cfg(tmp_path)
    cfg.compile = "auto"
    cfg.model_name = "skipped_model"
    monkeypatch.setattr(idist, "device", lambda: torch.device("cpu"))
    monkeypatch.setattr("gatle_ignite.enhancements.compile.resolve_compile", lambda _cfg: True)

    task = importlib.import_module(cfg.main_runner).Trainer(0, cfg)
    task.setup()

    assert not is_compiled(task.model), "auto compiled a model dynamo would skip"
    assert "dynamo would skip" in capsys.readouterr().out


def test_auto_does_not_change_the_trained_model(tmp_path):
    """`compile` must not advance the RNG, or cfg.seed gives different initial weights."""

    def weights(setting):
        cfg = _synthetic_cfg(tmp_path)
        cfg.compile = setting
        task = importlib.import_module(cfg.main_runner).Trainer(0, cfg)
        task.setup()
        return {k: v.clone() for k, v in task.model.state_dict().items()}

    baseline, auto = weights(False), weights("auto")
    assert set(baseline) == set(auto)
    for key, tensor in baseline.items():
        assert torch.equal(tensor, auto[key]), f"{key} differs: 'auto' moved the RNG"


def test_an_unknown_compile_value_is_refused(tmp_path):
    from gatle_ignite.enhancements.compile import resolve_compile

    cfg = _synthetic_cfg(tmp_path)
    cfg.compile = "yes please"
    with pytest.raises(ConfigError, match="must be True, False"):
        resolve_compile(cfg)


def test_auto_trains_end_to_end(tmp_path):
    cfg = _synthetic_cfg(tmp_path)
    cfg.compile = "auto"
    task = importlib.import_module(cfg.main_runner).Trainer(0, cfg)
    task.setup()
    assert not is_compiled(task.model), "auto said OFF on CPU, so nothing should be compiled"


def test_the_guard_fires_when_auto_resolved_ON_and_an_override_ignored_it(tmp_path, monkeypatch):
    """The only test that fails if the guard checks `cfg.compile is True` instead."""
    cfg = _synthetic_cfg(tmp_path)
    cfg.compile = "auto"
    monkeypatch.setattr("gatle_ignite.enhancements.compile.resolve_compile", lambda _cfg: True)
    base = importlib.import_module(cfg.main_runner).Trainer

    class Uncompiled(base):
        def build_model(self):
            from gatle_ignite.dispatch import import_entrypoint

            Model = import_entrypoint(self.cfg.model_name, "Model", field="model_name")
            return Model(**self.cfg.model_params)

    with pytest.raises(ConfigError, match="does not consider compiled"):
        Uncompiled(0, cfg).setup()
