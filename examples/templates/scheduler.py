"""TEMPLATE: an LR scheduler for cfg.lr_scheduler; lr_scheduler_params are splatted in."""

from ignite.engine import Events
from ignite.handlers import LRScheduler
from torch.optim.lr_scheduler import StepLR


def get_scheduler(cfg, train_dl, optimizer, engines, step_every_epochs=5, gamma=0.1):
    """Return (scheduler, engine_name, event); a list of schedulers works too.

    engine_name is any key of `engines`, whose value is None for an engine not built. An
    evaluator-driven scheduler is how a plateau schedule sees validation metrics.
    """
    # Wrap a torch scheduler in ignite's LRScheduler: it is attached as an event handler, so it
    # must be callable. Its first call applies the initial value, so N epochs give N-1 decays.
    scheduler = LRScheduler(StepLR(optimizer, step_size=step_every_epochs, gamma=gamma))
    return scheduler, "trainer", Events.EPOCH_COMPLETED
