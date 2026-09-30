import ignite.distributed as idist
import pytest
import torch

from gatle_ignite.dispatch import ConfigError
from gatle_ignite.utils.amp import autocast_device_type, resolve_amp_dtype


@pytest.mark.parametrize(
    "requested,expected",
    [("fp32", torch.float32), ("bf16", torch.bfloat16)],
)
def test_amp_dtype_can_be_forced(requested, expected):
    dtype, _ = resolve_amp_dtype(requested)
    assert dtype == expected


def test_fp16_resolves_on_cuda(monkeypatch):
    monkeypatch.setattr("gatle_ignite.utils.amp.idist.device", lambda: torch.device("cuda"))
    dtype, enabled = resolve_amp_dtype("fp16")
    assert dtype == torch.float16
    assert enabled is True


@pytest.mark.parametrize("device", ["cpu"])
def test_fp16_is_refused_off_cuda(monkeypatch, device):
    """Off CUDA there is no GradScaler, so fp16 would silently underflow."""
    monkeypatch.setattr("gatle_ignite.utils.amp.idist.device", lambda: torch.device(device))
    with pytest.raises(ConfigError, match="fp16"):
        resolve_amp_dtype("fp16")


def test_amp_auto_falls_back_to_fp32_without_a_capable_gpu():
    dtype, enabled = resolve_amp_dtype("auto")
    if idist.device().type == "cpu":
        assert dtype == torch.float32
        assert enabled is False


def test_autocast_device_type_follows_the_run_device(monkeypatch):
    monkeypatch.setattr("gatle_ignite.utils.amp.idist.device", lambda: torch.device("cuda"))
    assert autocast_device_type() == "cuda"
    monkeypatch.setattr("gatle_ignite.utils.amp.idist.device", lambda: torch.device("cpu"))
    assert autocast_device_type() == "cpu"


def test_unknown_amp_dtype_is_rejected():
    with pytest.raises(ConfigError, match="amp_dtype"):
        resolve_amp_dtype("float8")
