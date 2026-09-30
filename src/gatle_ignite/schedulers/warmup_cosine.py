"""Linear warmup, then cosine anneal to lr * lr_scale_factor."""

from ignite.engine import Events
from ignite.handlers import create_lr_scheduler_with_warmup
from torch.optim.lr_scheduler import CosineAnnealingLR

from gatle_ignite.schedulers._utils import base_lr, steps_per_epoch

MIN_WARMUP_EVENTS = 2


def get_scheduler(
    cfg,
    train_dl,
    optimizer,
    engines,
    warmup_epochs=1,
    warmup_start_factor=0.01,
    lr_scale_factor=0.01,
    **kwargs,
):
    """Factors are relative to the configured lr, so a config changes lr in one place."""
    per_epoch = steps_per_epoch(cfg, train_dl)
    total_steps = max(cfg.max_epochs * per_epoch, 1)
    lr = base_lr(cfg, optimizer)

    # ignite rejects a 1-event warmup, which is what a smoke run (train_length=1) gives.
    warmup_duration = max(int(warmup_epochs * per_epoch), MIN_WARMUP_EVENTS)

    cosine = CosineAnnealingLR(
        optimizer,
        T_max=max(total_steps - warmup_duration, 1),
        eta_min=lr * lr_scale_factor,
    )
    scheduler = create_lr_scheduler_with_warmup(
        cosine,
        warmup_start_value=lr * warmup_start_factor,
        warmup_end_value=lr,
        warmup_duration=warmup_duration,
        **kwargs,
    )
    return scheduler, "trainer", Events.ITERATION_STARTED
