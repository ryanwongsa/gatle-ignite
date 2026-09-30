"""Finding checkpoints on disk, and loading weights out of them.

The prefixes are what existing runs were saved under: renaming one orphans them.
"""

import re
from pathlib import Path

from gatle_ignite.dispatch import ConfigError

LATEST_PREFIX = "latest_epoch"
BEST_SUFFIX = "best_result"

# The LAST number in the name: valid_best_result_checkpoint_15_0.9258.pt -> 0.9258.
# That is the score only in best_result files; in any other it is the step.
_SCORE_RE = re.compile(r"_(-?\d+(?:\.\d+)?)\.[^.]+$")


def best_prefix_for(prefix=""):
    return f"{prefix}{BEST_SUFFIX}"


def _candidates(save_dir, prefix, extension="pt"):
    directory = Path(save_dir)
    if not directory.is_dir():
        return []
    return sorted(directory.glob(f"{prefix}*.{extension}"))


def score_of(path):
    """The score ignite embedded in a checkpoint filename, or None."""
    match = _SCORE_RE.search(Path(path).name)
    return float(match.group(1)) if match else None


def find_latest_checkpoint(save_dir, extension="pt"):
    """Newest `latest_epoch*` checkpoint by mtime, or None."""
    files = _candidates(save_dir, LATEST_PREFIX, extension)
    return max(files, key=lambda p: p.stat().st_mtime) if files else None


def find_best_checkpoint(save_dir, prefix="", extension="pt"):
    """Highest-scoring `{prefix}best_result*` checkpoint, or None.

    With n_saved > 1 several survive, and the newest is only the last to qualify.
    """
    files = _candidates(save_dir, best_prefix_for(prefix), extension)
    if not files:
        return None

    scored = [(score_of(f), f) for f in files]
    scored = [(s, f) for s, f in scored if s is not None]
    if not scored:
        # Only reachable for files this framework did not write: its own names end in a number.
        return max(files, key=lambda p: p.stat().st_mtime)

    # Ties broken by mtime: the later of two equal scores is the more recent model.
    return max(scored, key=lambda pair: (pair[0], pair[1].stat().st_mtime))[1]


class RunMeta:
    """Run settings recorded into the checkpoint, so a resume can detect a mismatch.

    A scheduler's state_dict carries its geometry (a cosine's T_max), so resuming under a new
    max_epochs silently replays the old schedule. Recording max_epochs lets resume warn.
    """

    def __init__(self, max_epochs):
        self.max_epochs = max_epochs
        self.restored_max_epochs = None

    def state_dict(self):
        return {"max_epochs": self.max_epochs}

    def load_state_dict(self, state_dict):
        self.restored_max_epochs = state_dict.get("max_epochs", None)


def resolve_eval_checkpoint(save_dir, load_from_ckpt="best", prefix="valid_"):
    if load_from_ckpt == "latest":
        return find_latest_checkpoint(save_dir)
    return find_best_checkpoint(save_dir, prefix=prefix)


def _unwrap(blob):
    """A framework checkpoint's "model", or the blob itself when it is a bare state dict."""
    return blob["model"] if isinstance(blob, dict) and "model" in blob else blob


def _align_module_prefix(state, wanted):
    """Add or strip DDP's `module.` prefix so a checkpoint fits the module in hand."""
    if not isinstance(state, dict) or not state or not wanted:
        return state
    keys = [k for k in state if isinstance(k, str)]
    if not keys or set(keys) & set(wanted):
        return state

    ddp_ckpt = all(k.startswith("module.") for k in keys)
    ddp_model = all(k.startswith("module.") for k in wanted)
    if ddp_ckpt and not ddp_model:
        return {k[len("module.") :]: v for k, v in state.items()}
    if ddp_model and not ddp_ckpt:
        return {f"module.{k}": v for k, v in state.items()}
    return state


def report_weight_match(state, wanted, path):
    """Say how many tensors a load matched; raise if none did.

    Under strict=False a 0% match loads nothing and warns about nothing.
    """
    if not isinstance(state, dict):
        return
    wanted = set(wanted)
    matched = wanted & set(state)
    if matched:
        print(f"LOADED {len(matched)}/{len(wanted)} tensors")
        return

    where = path if isinstance(path, (str, Path)) else "an in-memory checkpoint"
    raise ConfigError(
        f"loaded 0 of {len(wanted)} tensors from {where}: the model expects e.g. "
        f"{sorted(wanted)[:1]}, the checkpoint offers e.g. "
        f"{sorted(k for k in state if isinstance(k, str))[:1]}.\n"
        f"  Load a checkpoint saved from this model, or rename its keys to match."
    )


def load_weights(source, module, strict=True, map_location="cpu"):
    """Load a checkpoint's model weights into `module`. Returns the load result.

    For any model but the one `model_name` names: a distillation teacher, a frozen encoder.
    `source` is a path or a loaded blob, full checkpoint or bare state dict.
    """
    import torch

    blob = (
        torch.load(source, map_location=map_location) if isinstance(source, (str, Path)) else source
    )
    state = _unwrap(blob)
    wanted = list(module.state_dict())
    state = _align_module_prefix(state, wanted)
    report_weight_match(state, wanted, source)
    return module.load_state_dict(state, strict=strict)
