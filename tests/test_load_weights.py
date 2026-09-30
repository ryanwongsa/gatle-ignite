"""load_weights: the loader behind model_checkpoint_dir, and for any second model."""

import pytest
import torch
import torch.nn as nn

from gatle_ignite import ConfigError, load_weights


def _net(seed=0):
    torch.manual_seed(seed)
    return nn.Sequential(nn.Linear(4, 3), nn.ReLU(), nn.Linear(3, 2))


def _differs(a, b):
    return any(not torch.equal(a[k], b[k]) for k in a)


def test_loads_from_a_full_framework_checkpoint(tmp_path):
    source, target = _net(0), _net(1)
    assert _differs(source.state_dict(), target.state_dict())

    path = tmp_path / "ckpt.pt"
    torch.save({"model": source.state_dict(), "optimizer": {}, "trainer": {}}, path)
    load_weights(path, target)

    for key, value in source.state_dict().items():
        assert torch.equal(target.state_dict()[key], value)


def test_loads_from_a_bare_state_dict(tmp_path):
    source, target = _net(0), _net(1)
    path = tmp_path / "bare.pt"
    torch.save(source.state_dict(), path)
    load_weights(path, target)
    assert not _differs(source.state_dict(), target.state_dict())


def test_loads_an_already_loaded_blob():
    source, target = _net(0), _net(1)
    load_weights({"model": source.state_dict()}, target)
    assert not _differs(source.state_dict(), target.state_dict())


def test_a_ddp_saved_checkpoint_loads_into_a_plain_module():
    """The `module.` prefix is mechanical, so it is aligned rather than reported."""
    source, target = _net(0), _net(1)
    ddp_style = {f"module.{k}": v for k, v in source.state_dict().items()}
    load_weights({"model": ddp_style}, target)
    assert not _differs(source.state_dict(), target.state_dict())


def test_a_plain_checkpoint_loads_into_a_ddp_wrapped_module():

    class FakeDDP(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

    source = _net(0)
    target = FakeDDP(_net(1))
    load_weights({"model": source.state_dict()}, target)
    assert not _differs(source.state_dict(), target.module.state_dict())


def test_a_checkpoint_that_matches_nothing_raises(tmp_path):
    """Genuinely different names, not a prefix: loading zero tensors is never intended."""
    source, target = _net(0), _net(1)
    mangled = {f"encoder.{k}": v for k, v in source.state_dict().items()}
    before = {k: v.clone() for k, v in target.state_dict().items()}

    with pytest.raises(ConfigError, match="loaded 0 of"):
        load_weights({"model": mangled}, target)
    assert not _differs(before, target.state_dict()), "it half-loaded on the way out"


def test_a_partial_match_loads_and_is_reported(tmp_path, capsys):
    source, target = _net(0), _net(1)
    state = source.state_dict()
    dropped = sorted(state)[0]
    partial = {k: v for k, v in state.items() if k != dropped}

    load_weights({"model": partial}, target, strict=False)

    out = capsys.readouterr().out
    assert f"LOADED {len(partial)}/{len(state)} tensors" in out, out


def test_strict_still_means_strict():
    source, target = _net(0), _net(1)
    partial = {k: v for k, v in source.state_dict().items() if k != sorted(source.state_dict())[0]}
    with pytest.raises(RuntimeError, match="Missing key"):
        load_weights({"model": partial}, target, strict=True)


def test_a_second_model_is_the_point():
    teacher_ckpt = {"model": _net(0).state_dict(), "ema": {"shadow": {}}}
    teacher = _net(9)
    load_weights(teacher_ckpt, teacher)
    for param in teacher.parameters():
        param.requires_grad_(False)
    teacher.eval()

    assert not _differs(_net(0).state_dict(), teacher.state_dict())
    assert not any(p.requires_grad for p in teacher.parameters())
