"""Drop the LR when a validation metric stops improving."""

from ignite.engine import Events
from ignite.handlers import ReduceLROnPlateauScheduler

from gatle_ignite.dispatch import ConfigError


class _CheckpointablePlateau(ReduceLROnPlateauScheduler):
    # ignite binds its own _reduce_lr onto the torch scheduler, and torch's state_dict
    # copies it: a checkpoint holding a bound method fails torch.load(weights_only=True).
    def state_dict(self):
        state = super().state_dict()
        state["scheduler"].pop("_reduce_lr", None)
        return state


def get_scheduler(
    cfg,
    train_dl,
    optimizer,
    engines,
    metric_name=None,
    mode="min",
    factor=0.1,
    patience=10,
    **kwargs,
):
    metric_name = metric_name or cfg.get("score_name", None)
    if not metric_name:
        raise ConfigError(
            "plateau scheduler has no metric to watch.\n"
            "  Set lr_scheduler_params.metric_name (or cfg.score_name) to one of "
            "cfg.val_metrics, e.g. 'valid/acc': eval steps compute no loss."
        )
    scheduler = _CheckpointablePlateau(
        optimizer,
        metric_name=metric_name,
        mode=mode,
        factor=factor,
        patience=patience,
        **kwargs,
    )
    return scheduler, "evaluator", Events.COMPLETED
