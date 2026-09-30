import ignite.distributed as idist
from ignite.utils import convert_tensor

from gatle_ignite.dispatch import ConfigError


def to_device(batch, device=None, non_blocking=True):
    """Move a (possibly nested) batch structure to the active device.

    ignite's convert_tensor, so a prep_batch need not import idist to name the device.
    """
    return convert_tensor(
        batch,
        device=device if device is not None else idist.device(),
        non_blocking=non_blocking,
    )


def get_value(container, name):
    """get_value(y, "logits") or get_value(x, ("targets", "labels")).

    A string is a top-level key and a tuple or list is a path, which is what lets a config
    wire a loss or metric by naming its `src_name` and `tgt_name`.
    """
    if isinstance(name, (tuple, list)):
        value = container
        for i, key in enumerate(name):
            try:
                value = value[key]
            except (KeyError, TypeError) as e:
                path = " -> ".join(map(str, name[: i + 1]))
                raise ConfigError(f"could not resolve path {tuple(name)!r} at {path!r}: {e}") from e
        return value
    try:
        return container[name]
    except (KeyError, TypeError) as e:
        available = list(container) if hasattr(container, "keys") else type(container)
        raise ConfigError(f"could not resolve {name!r}; available: {available}") from e
