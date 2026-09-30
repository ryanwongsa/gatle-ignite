"""Loading a config module from a file path, and applying CLI overrides."""

import ast
import importlib.util
from pathlib import Path

from gatle_ignite.dispatch import ConfigError


def load_config_from_file(config_file_path):
    """Import a .py file by path and call its get_config()."""
    path = Path(config_file_path)
    if not path.exists():
        raise ConfigError(f"config file not found: {path}")

    spec = importlib.util.spec_from_file_location("gatle_config_module", str(path))
    if spec is None or spec.loader is None:
        raise ConfigError(f"could not load a python module from: {path}")
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except ImportError as e:
        # A ConfigError prints the one line to fix, not an importlib traceback.
        raise ConfigError(_import_error_message(path, e)) from e

    if not hasattr(module, "get_config"):
        raise ConfigError(f"config file {path} must define get_config()")
    return module.get_config()


def _import_error_message(path, error):
    if "relative import" in str(error):
        return (
            f"{path} uses a relative import, which a config cannot: it is loaded by file path.\n"
            f"  Use an absolute import, `from dataloaders.vocab import X` not `from .vocab import X`."
        )
    return (
        f"could not import {path}: {error}\n"
        f"  Run from the directory its dotted paths are relative to: the CWD is put on "
        f"sys.path, and a config must use absolute imports."
    )


def apply_overrides(cfg, overrides):
    """Apply literal-parsed `key=value` overrides; a dotted key reaches a nested field.

    Only existing keys, at every level: a typo must not add a dead key and train on the default.
    """
    for item in overrides or []:
        if "=" not in item:
            raise ConfigError(f"override {item!r} is not of the form key=value")
        key, raw = item.split("=", 1)
        key = key.strip()
        try:
            value = ast.literal_eval(raw)
        except (ValueError, SyntaxError):
            value = raw

        target, leaf = _resolve_override_path(cfg, key)
        target[leaf] = value
    return cfg


def _is_mapping(value):
    return hasattr(value, "keys") and hasattr(value, "__getitem__")


def _resolve_override_path(cfg, key):
    """Walk a dotted key to (container, leaf), creating nothing along the way."""
    parts = key.split(".")
    target = cfg

    for depth, part in enumerate(parts[:-1]):
        so_far = ".".join(parts[: depth + 1])
        if part not in target:
            raise ConfigError(f"cannot override unknown config field {so_far!r}")
        target = target[part]
        if not _is_mapping(target):
            # Name the segment at fault; the `in` check below would blame the value's type.
            raise ConfigError(
                f"cannot override {key!r}: {so_far!r} is a {type(target).__name__}, "
                f"not a group of fields"
            )

    leaf = parts[-1]
    if leaf not in target:
        if len(parts) == 1:
            raise ConfigError(f"cannot override unknown config field {leaf!r}")
        raise ConfigError(
            f"cannot override unknown config field {key!r}: "
            f"{'.'.join(parts[:-1])!r} has no {leaf!r}. It holds: {sorted(target.keys())}"
        )
    return target, leaf
