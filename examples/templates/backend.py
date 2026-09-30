"""TEMPLATE: a logging sink, enabled by a dotted entry in cfg.logger_name. This one writes a CSV."""

import csv
from pathlib import Path

from gatle_ignite.callbacks.logging import Backend as BaseBackend


class Backend(BaseBackend):
    """The name `Backend` is what the framework imports.

    Subclass the base: the framework calls `watch` and `finish` unguarded, and it has both.
    """

    def __init__(self, cfg):
        super().__init__(cfg)
        # Third-party imports go in here, not at module top, so only runs using this sink pay.
        self.path = Path(cfg.save_dir) / f"{cfg.name}_metrics.csv"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._columns = None

    def log(self, metrics, step=None, epoch=None):
        """`metrics` is flat {str: float}, keyed by engine ("valid/top2").

        `step` or `epoch` may be None, depending on which event fired: never assume both.
        """
        row = {"step": step, "epoch": epoch, **metrics}
        # Keys vary by event, so the first row fixes the header and later extra keys are dropped.
        write_header = self._columns is None
        if write_header:
            self._columns = list(row)
        with self.path.open("a", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=self._columns, extrasaction="ignore")
            if write_header:
                writer.writeheader()
            writer.writerow(row)

    def watch(self, model):
        """Called once per run, only when cfg.watch_grad is set. Usually a no-op."""

    def finish(self, failed=False):
        """Called at the end of EVERY run. `failed` is True if training raised."""
