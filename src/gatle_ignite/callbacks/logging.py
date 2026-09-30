"""Fans metrics out to every backend named in cfg.logger_name."""

import ignite.distributed as idist
import torch
from ignite.engine import Events

from gatle_ignite.dispatch import import_entrypoint

BUILTIN_BACKENDS = ("text", "wandb", "discord")
NOT_A_BACKEND = ("pbar",)


class Backend:
    """A logging sink. Only backends named in cfg.logger_name are imported."""

    def __init__(self, cfg):
        self.cfg = cfg

    def log(self, metrics, step=None, epoch=None):
        """Record a flat {name: scalar} mapping."""

    def watch(self, model):
        """Optionally track gradients/parameters."""

    def finish(self, failed=False):
        pass


class LoggingCallback:
    def __init__(self, cfg):
        self.cfg = cfg
        self.backends = []

    def start(self):
        # Rank 0 logs; other ranks would duplicate every series.
        if idist.get_rank() != 0:
            return self
        for name in self.cfg.logger_name:
            if name in NOT_A_BACKEND:
                continue
            path = f"gatle_ignite.callbacks.backends.{name}" if name in BUILTIN_BACKENDS else name
            self.backends.append(import_entrypoint(path, "Backend", field="logger_name")(self.cfg))
        return self

    def _log(self, metrics, step=None, epoch=None):
        for backend in self.backends:
            backend.log(metrics, step=step, epoch=epoch)

    def on_train_epoch_end(self, trainer, optimizer):
        @trainer.on(Events.EPOCH_COMPLETED(every=1))
        def log_train(engine):
            self._log(_flatten(engine.state.metrics), epoch=engine.state.epoch)

        @trainer.on(Events.ITERATION_COMPLETED(every=self.cfg.log_every))
        def log_lr(engine):
            lrs = {f"train/lr_{i}": pg["lr"] for i, pg in enumerate(optimizer.param_groups)}
            self._log(lrs, step=engine.state.iteration, epoch=engine.state.epoch)

    def on_train_iteration(self, trainer, model):
        if not self.cfg.watch_grad or not self.backends:
            return

        for backend in self.backends:
            backend.watch(model)

        @trainer.on(Events.ITERATION_COMPLETED(every=self.cfg.log_every))
        def log_grad_norm(engine):
            total = 0.0
            for p in model.parameters():
                if p.grad is not None:
                    total += p.grad.detach().float().norm(2).item() ** 2
            self._log(
                {"train/grad_norm": total**0.5},
                step=engine.state.iteration,
                epoch=engine.state.epoch,
            )

    def on_valid_epoch_end(self, trainer, engine):
        @engine.on(Events.EPOCH_COMPLETED(every=1))
        def log_eval(eval_engine):
            self._log(_flatten(eval_engine.state.metrics), epoch=trainer.state.epoch)

    def on_completion(self, trainer):
        @trainer.on(Events.COMPLETED)
        def done(engine):
            print(f"TRAINING COMPLETE after {engine.state.epoch} epochs")

    def finish(self, failed=False):
        for backend in self.backends:
            backend.finish(failed=failed)
        self.backends = []


def _flatten(metrics):
    flat = {}
    for key, value in metrics.items():
        if "summary" in key:
            continue
        if isinstance(value, dict):
            for sub, sub_value in value.items():
                flat[f"{key}_{sub}"] = _scalar(sub_value)
        else:
            flat[key] = _scalar(value)
    return flat


def _scalar(value):
    if torch.is_tensor(value):
        return value.item() if value.numel() == 1 else value.tolist()
    return value
