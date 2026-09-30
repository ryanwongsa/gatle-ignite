"""Mixed-precision dtype, probed on the run's device rather than read off a GPU-name list."""

import ignite.distributed as idist
import torch

from gatle_ignite.dispatch import ConfigError

_CHOICES = ("auto", "bf16", "fp16", "fp32")


def resolve_amp_dtype(amp_dtype="auto"):
    """Return (torch dtype, autocast_enabled) for the requested precision."""
    if amp_dtype not in _CHOICES:
        raise ConfigError(f"amp_dtype must be one of {_CHOICES}, got {amp_dtype!r}")

    if amp_dtype == "auto":
        device_type = idist.device().type
        # is_bf16_supported() is CUDA-specific and safe to call only once cuda is confirmed.
        if device_type == "cuda" and torch.cuda.is_bf16_supported():
            return torch.bfloat16, True
        return torch.float32, False

    if amp_dtype == "fp32":
        return torch.float32, False
    if amp_dtype == "bf16":
        return torch.bfloat16, True

    # fp16 underflows without a GradScaler, and build_optimizer creates one only on CUDA,
    # so fp16 anywhere else would train silently on underflowed gradients.
    device_type = idist.device().type
    if device_type != "cuda":
        raise ConfigError(
            f"amp_dtype='fp16' needs a GradScaler, which exists only on CUDA; this run is on "
            f"{device_type!r}.\n  Use 'bf16', or 'auto' to pick the right dtype for the device."
        )
    return torch.float16, True


def autocast_device_type():
    return idist.device().type
