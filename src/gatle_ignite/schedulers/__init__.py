from gatle_ignite.dispatch import ConfigError, import_entrypoint


def prep_scheduler(lr_scheduler, cfg, train_dl, optimizer, engines):
    """-> (scheduler, engine_name, event_name). A scheduler of None means no scheduling."""
    if lr_scheduler is None:
        return None, "trainer", None
    get_scheduler = import_entrypoint(lr_scheduler, "get_scheduler", field="lr_scheduler")
    params = dict(cfg.get("lr_scheduler_params", {}) or {})
    try:
        result = get_scheduler(cfg, train_dl, optimizer, engines, **params)
    except TypeError as e:
        if "unexpected keyword argument" in str(e):
            raise ConfigError(
                f"lr_scheduler_params not accepted by {lr_scheduler}.get_scheduler: {e}"
            ) from e
        raise

    if not isinstance(result, tuple) or len(result) != 3:
        raise ConfigError(
            f"{lr_scheduler}.get_scheduler must return "
            f"(scheduler, engine_name, event_name), got {result!r}"
        )
    return result


def attach_scheduler(scheduler, engine_name, event_name, engines):
    if not scheduler:
        return scheduler
    if engine_name not in engines:
        raise ConfigError(
            f"scheduler engine_name must be one of {sorted(engines)}, got {engine_name!r}"
        )
    engine = engines[engine_name]
    if engine is None:
        raise ConfigError(
            f"scheduler targets the {engine_name!r} engine, but it was not built "
            f"(no dataset configured for it, or its cadence is 0?)"
        )
    for sch in scheduler if isinstance(scheduler, list) else [scheduler]:
        engine.add_event_handler(event_name, sch)
    return scheduler
