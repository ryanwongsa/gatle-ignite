"""Multiply the LR by gamma every step_size epochs."""

from ignite.engine import Events
from ignite.handlers import LRScheduler
from torch.optim.lr_scheduler import StepLR


def get_scheduler(cfg, train_dl, optimizer, engines, step_size=10, gamma=0.1, **kwargs):
    scheduler = LRScheduler(StepLR(optimizer, step_size=step_size, gamma=gamma, **kwargs))
    return scheduler, "trainer", Events.EPOCH_COMPLETED
