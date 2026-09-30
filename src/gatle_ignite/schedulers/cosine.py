"""Cosine anneal the LR every iteration over the whole run."""

from ignite.engine import Events
from ignite.handlers import CosineAnnealingScheduler

from gatle_ignite.schedulers._utils import base_lr, steps_per_epoch


def get_scheduler(
    cfg,
    train_dl,
    optimizer,
    engines,
    start_value=None,
    end_value=0.0,
    cycle_epochs=None,
    **kwargs,
):
    cycle_size = (cycle_epochs or cfg.max_epochs) * steps_per_epoch(cfg, train_dl)

    scheduler = CosineAnnealingScheduler(
        optimizer,
        "lr",
        start_value=start_value if start_value is not None else base_lr(cfg, optimizer),
        end_value=end_value,
        cycle_size=max(cycle_size, 2),
        **kwargs,
    )
    return scheduler, "trainer", Events.ITERATION_STARTED
