"""Print metrics to stdout. The default."""

import pprint

from gatle_ignite.callbacks.logging import Backend as BaseBackend


class Backend(BaseBackend):
    def log(self, metrics, step=None, epoch=None):
        header = f"epoch {epoch}" if epoch is not None else f"step {step}"
        print(f"--- {header} ---")
        pprint.pprint(metrics)
