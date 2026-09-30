"""Weighted sum of sub-losses, built from config. Each sub-loss returns a bare tensor."""

import torch
import torch.nn as nn

from gatle_ignite.dispatch import ConfigError, import_entrypoint


class Loss(nn.Module):
    def __init__(self, dict_of_loss_params):
        super().__init__()
        if not dict_of_loss_params:
            raise ConfigError(
                "criterion_params.dict_of_loss_params is empty.\n"
                "  Add at least one sub-loss entry, with a cls_name and a weight."
            )

        # Not a dict: a sub-loss's buffers must follow .to(device) and appear in state_dict().
        self.dict_loss_functions = nn.ModuleDict()
        self.weighting = {}
        self.gating = {}

        for key, entry in dict_of_loss_params.items():
            for required in ("cls_name", "weight"):
                if required not in entry:
                    raise ConfigError(
                        f"loss entry {key!r} is missing {required!r}.\n"
                        f"  Add {required!r} to dict_of_loss_params[{key!r}]."
                    )
            loss_cls = import_entrypoint(
                entry["cls_name"], "Loss", field=f"dict_of_loss_params.{key}.cls_name"
            )
            self.dict_loss_functions[key] = loss_cls(**entry.get("loss_params", {}))
            self.weighting[key] = entry["weight"]
            self.gating[key] = (
                entry.get("start_iteration", None),
                entry.get("end_iteration", None),
            )

        # BaseTrainer builds a "loss_{key}_avg" metric per entry from this.
        self.crit_keys = list(dict_of_loss_params.keys())

    def _is_active(self, key, iteration):
        start, end = self.gating[key]
        if iteration is None or (start is None and end is None):
            return True
        if start is not None and iteration < start:
            return False
        if end is not None and iteration > end:
            return False
        return True

    def forward(self, y_pred, target, iteration=None):
        total = None
        dict_loss = {}
        for key, loss_fn in self.dict_loss_functions.items():
            if not self._is_active(key, iteration):
                # Keep the key present so the metric series stays continuous.
                dict_loss[f"loss_{key}"] = torch.zeros((), device=_device_of(y_pred))
                continue
            value = self.weighting[key] * loss_fn(y_pred, target, iteration)
            dict_loss[f"loss_{key}"] = value
            total = value if total is None else total + value

        if total is None:
            # A constant here would fail later in backward(), with an opaque grad_fn error.
            raise ConfigError(
                f"all loss terms are gated off at iteration {iteration}: "
                f"{ {k: self.gating[k] for k in self.crit_keys} }.\n"
                f"  Widen a term's start_iteration/end_iteration so one is always active."
            )
        return total, dict_loss


def _device_of(y_pred):
    for v in y_pred.values() if hasattr(y_pred, "values") else []:
        if torch.is_tensor(v):
            return v.device
    return torch.device("cpu")
