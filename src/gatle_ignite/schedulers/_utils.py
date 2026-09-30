from gatle_ignite.dispatch import ConfigError


def steps_per_epoch(cfg, train_dl):
    return cfg.get("train_length", None) or len(train_dl)


def base_lr(cfg, optimizer):
    """The config's lr if it sets one, else the optimizer's own default."""
    lr = cfg.optimizer_params.get("lr", None) if "optimizer_params" in cfg else None
    if lr is not None:
        return lr
    if optimizer.param_groups:
        return optimizer.param_groups[0]["lr"]
    raise ConfigError("cannot determine a base learning rate: optimizer has no param groups")
