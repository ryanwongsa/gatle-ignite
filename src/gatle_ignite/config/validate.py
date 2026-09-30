"""Config checks. Project fields go unchecked: the framework does not read them."""

import warnings

from gatle_ignite import dispatch
from gatle_ignite.config.base import REQUIRED_FIELDS
from gatle_ignite.dispatch import ConfigError, import_entrypoint
from gatle_ignite.enhancements.compile import check_compile_value
from gatle_ignite.trainer.specs import EVAL_SPECS


def validate_config(cfg, specs=EVAL_SPECS):
    """Raise on a missing required field or a bad value; warn when a score can never be kept.

    `specs` is the trainer's `eval_specs()`, so its own engines get the scoring warnings too.
    """
    missing = [f for f in REQUIRED_FIELDS if cfg.get(f, None) in (None, "")]
    if missing:
        raise ConfigError(
            "config is missing required field(s): "
            + ", ".join(repr(m) for m in missing)
            + "\n  Start from gatle_ignite.base_config() and set them in get_config()."
        )

    _check_n_saved(cfg)
    check_compile_value(cfg)

    for spec in specs:
        _warn_about_scoring(cfg, spec)
    return cfg


def _check_n_saved(cfg):
    """`cfg.n_saved` reaches ignite's Checkpoint, which rejects it below 1 mid-run."""
    if not cfg.get("save_ckpt", True):
        return
    n_saved = cfg.get("n_saved", 1)
    if n_saved is None:
        return
    if not isinstance(n_saved, int) or isinstance(n_saved, bool) or n_saved < 1:
        raise ConfigError(
            f"cfg.n_saved must be an int >= 1, or None to keep every checkpoint, got {n_saved!r}.\n"
            f"  To write no checkpoints at all, set cfg.save_ckpt = False."
        )


def _warn_about_scoring(cfg, spec):
    if spec.score_field is None:
        return
    score_name = cfg.get(spec.score_field, None)
    if not score_name:
        return

    factor = cfg.get(spec.score_factor_field, 1)
    if factor not in (1, -1):
        warnings.warn(
            f"{spec.score_factor_field}={factor!r}; checkpointing keeps the MAXIMUM score, "
            "so use -1 for metrics where lower is better (e.g. loss, CER).",
            stacklevel=3,
        )

    if spec.every_field and cfg.get(spec.every_field, None) == 0:
        warnings.warn(
            f"{spec.score_field} is set but {spec.every_field}=0, so no {spec.key} engine is "
            f"built and no best-on-{spec.engine_type} checkpoint will ever be written, only "
            f"'latest'. Set {spec.every_field} >= 1, or clear {spec.score_field}.",
            stacklevel=3,
        )
    if not cfg.get(f"{spec.split}_ds_name", None):
        warnings.warn(
            f"{spec.score_field} is set but there is no {spec.split}_ds_name, so no {spec.key} "
            f"engine is built and no best-on-{spec.engine_type} checkpoint will ever be "
            f"written, only 'latest'.",
            stacklevel=3,
        )


def _check(errors, path, attr, field):
    try:
        import_entrypoint(path, attr, field=field)
    except ConfigError as e:
        errors.append(str(e))


def check_dispatch(cfg):
    """Raise ConfigError listing every dotted path in `cfg` that does not resolve."""
    errors = []

    for _label, attr, _contract, fields in dispatch.CONTRACTS:
        for field in fields:
            path = cfg.get(field, None)
            if path:
                _check(errors, path, attr, field)

    dict_of_loss_params = (cfg.get("criterion_params", {}) or {}).get(
        "dict_of_loss_params", {}
    ) or {}
    for key, entry in dict_of_loss_params.items():
        if entry.get("cls_name", None):
            _check(errors, entry["cls_name"], "Loss", f"dict_of_loss_params.{key}.cls_name")

    for field in ("train_metrics", "val_metrics", "tester_metrics"):
        spec = cfg.get(field, {}) or {}
        if not hasattr(spec, "items"):
            continue
        for key, entry in spec.items():
            if entry.get("cls_name", None):
                _check(errors, entry["cls_name"], "get_metric", f"{field}.{key}.cls_name")

    for field in ("train_ds_params", "valid_ds_params", "test_ds_params"):
        params = cfg.get(field, {}) or {}
        sampler = params.get("sampler_params", None) if hasattr(params, "get") else None
        if sampler and sampler.get("cls_name", None):
            _check(errors, sampler["cls_name"], "get_sampler", f"{field}.sampler_params.cls_name")

    # Logger backends. Short names are the one exception to the dotted-path rule.
    from gatle_ignite.callbacks.logging import BUILTIN_BACKENDS, NOT_A_BACKEND

    for name in cfg.get("logger_name", []) or []:
        if name in NOT_A_BACKEND or name in BUILTIN_BACKENDS:
            continue
        _check(errors, name, "Backend", "logger_name")

    if errors:
        raise ConfigError(
            f"{len(errors)} dotted path(s) in this config do not resolve:\n\n"
            + "\n\n".join(f"  - {e}" for e in errors)
        )
    return cfg
