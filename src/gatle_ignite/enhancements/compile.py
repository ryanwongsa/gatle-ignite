"""`torch.compile` as a config field.

`model.compile()` in place, never `torch.compile(model)`: ignite's Checkpoint keeps its
`_orig_mod.` key prefix. Dynamo skips frames defined inside torch, DDP's forward included,
so the skip check runs on the unwrapped model.
"""

import ignite.distributed as idist
import torch

from gatle_ignite.dispatch import ConfigError

_AUTO = "auto"


def _would_be_skipped(model):
    """True if dynamo will skip this model's forward, making a compile a silent no-op.

    `trace_rules` is private torch: if it moves, answer False and run eager rather than crash.
    """
    try:
        from torch._dynamo.trace_rules import check
    except ImportError:
        return False
    return bool(check(type(model).forward))


def compile_model(model, compile_params=None):
    """Compile `model` IN PLACE and return the same object.

    A `build_model` override calls this itself, before any `idist.auto_model` wrap.
    """
    if _would_be_skipped(model):
        raise ConfigError(
            f"cfg.compile is set, but dynamo skips {type(model).__module__}."
            f"{type(model).__name__} (defined inside torch), so nothing would compile.\n"
            f"  Wrap it in your own nn.Module subclass, or leave cfg.compile unset."
        )

    params = dict(compile_params or {})
    model.compile(**params)
    return model


def is_compiled(module):
    """True if `module`, or the module a wrapper holds, was compiled in place.

    Never walks submodules: a compiled encoder under an eager forward still trains eager.
    """
    for _ in range(8):  # bounded: a wrapper chain, not a general graph walk
        if getattr(module, "_compiled_call_impl", None) is not None:
            return True
        inner = getattr(module, "module", None)
        if inner is None or inner is module or not hasattr(inner, "state_dict"):
            return False
        module = inner
    return False


def check_compile_value(cfg):
    """Raise unless `cfg.compile` is True, False, "auto" or unset.

    The field is placeholder(object) to hold "auto", which means ml_collections checks nothing.
    """
    setting = cfg.get("compile", None)
    if setting is None or isinstance(setting, bool) or setting == _AUTO:
        return
    raise ConfigError(f'cfg.compile must be True, False or "auto", got {setting!r}.')


def resolve_compile(cfg):
    """Decide whether to compile. "auto" never builds a model: that would move the global RNG."""
    check_compile_value(cfg)
    setting = cfg.get("compile", False)
    # None is the unset default: a field typed bool by its first value could not hold "auto".
    if setting is None or setting is False:
        return False
    if setting is True:
        return True

    if idist.device().type == "cpu":
        print("[compile] auto -> OFF (no accelerator; compiling CPU training rarely pays)")
        return False
    print(f"[compile] auto -> ON ({idist.device().type}, torch {torch.__version__})")
    return True


def compile_built_model(model, cfg):
    """Compile the freshly built model if the config asks. -> whether it did

    Call it before any DDP wrap, which dynamo skips.
    """
    compiling = resolve_compile(cfg)
    if compiling and cfg.get("compile", None) == _AUTO and _would_be_skipped(model):
        print(
            f"[compile] auto -> OFF ({type(model).__name__} is defined inside torch, "
            f"so dynamo would skip it and compile nothing)"
        )
        compiling = False
    if compiling:
        compile_model(model, cfg.compile_params)
    return compiling


def check_build_model_override(model, compiling, cfg):
    """Refuse a `build_model` override that skipped compile.

    `compiling` is what the default `build_model` decided, or None if it never ran.
    """
    if compiling is None:
        compiling = resolve_compile(cfg)
    if compiling and not is_compiled(model):
        raise ConfigError(
            "cfg.compile is set but build_model() returned a model this framework does not "
            "consider compiled; torch.compile(model) and self.encoder.compile() do not count.\n"
            "  Call gatle_ignite.enhancements.compile.compile_model(model) or model.compile() "
            "on the model you return, before any DDP wrap."
        )
