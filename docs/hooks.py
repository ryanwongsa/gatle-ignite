"""Renders each example page's spine: which of the nine build steps it changes.

A `<!-- spine: gan/gan_v0 -->` marker in an example page is replaced at build time by a
table read from that example's live config and Trainer class. Configs name their
components through f-strings and local variables, so this imports them rather than
parsing them.
"""

import re
import sys
from pathlib import Path

ROOT_DIR = str(Path(__file__).resolve().parent.parent)


def _on_path():
    """The repo root, so `examples.*` imports.

    Re-checked on every call rather than set once at import: mkdocstrings manages sys.path
    for its own `paths` setting, and an entry added at import time is gone by the time a
    page renders.
    """
    if ROOT_DIR not in sys.path:
        sys.path.insert(0, ROOT_DIR)


_on_path()

import importlib  # noqa: E402

from gatle_ignite import base_config  # noqa: E402
from gatle_ignite.trainer.base import BaseTrainer  # noqa: E402

BASE_METHODS = {n for n in dir(BaseTrainer) if not n.startswith("__")}

# Every spine is measured against this one, so it has no spine of its own.
REF = "examples.synthetic.configs.synthetic_v0"

# Shape fields: a difference here changes what the task is. Fields absent from this map
# differ on nearly every example without saying anything -- epoch counts, learning rates,
# and which module a dataset happens to live in.
SHAPE = {
    "aug_name": "dataset",
    "test_ds_name": "dataset",
    "model_name": "model",
    "criterion_name": "loss",
    "val_metrics": "metrics",
    "tester_metrics": "metrics",
    "every_test": "metrics",
    "score_name": "metrics",
    "score_factor": "metrics",
    "tester_score_name": "metrics",
    "tester_score_factor": "metrics",
    "optimizer_name": "optimizer",
    "lr_scheduler": "optimizer",
    "resume": "checkpoints",
    "n_saved": "checkpoints",
    "logger_name": "logging",
    "watch_grad": "logging",
    "amp_dtype": "running",
    "compile": "running",
    "accum_steps": "running",
    "grad_clip_norm": "running",
    "grad_clip_value": "running",
}

STEP_OF_METHOD = {
    "prep_batch": "prep-batch",
    "train_step": "prep-batch",
    "eval_step": "prep-batch",
    "forward": "prep-batch",
    "train_spec": "prep-batch",
    "eval_specs": "prep-batch",
    "build_model": "prep-batch",
    "build_dataloaders": "prep-batch",
    "backward": "prep-batch",
    "eval_context": "prep-batch",
    "dict_metric_from_list": "metrics",
    "extra_to_save": "checkpoints",
}

ORDER = [
    "dataset",
    "model",
    "prep-batch",
    "loss",
    "metrics",
    "optimizer",
    "checkpoints",
    "logging",
    "running",
]
TITLE = dict(
    zip(
        ORDER,
        "dataset model prep_batch loss metrics optimizer checkpoints logging running".split(),
    )
)

NOTHING = (
    "**Nothing beyond [the walk](../steps/dataset.md).** Its own dataset, model and "
    "`prep_batch`, and not one step that needs more than the walk already covers."
)
MARKER = re.compile(r"^<!--\s*spine:\s*([\w.-]+)/([\w.-]+)\s*-->$", re.M)


def _metric_modules(entries):
    """The `name (module)` pairs inside a metrics dict.

    These are ConfigDict, not dict: `isinstance(v, dict)` is False and `str(v)` is a YAML
    dump that would destroy the table.
    """
    out = []
    for key in sorted(entries.keys()):
        entry = entries[key]
        if hasattr(entry, "keys") and "cls_name" in entry:
            out.append(f"`{key}` ({str(entry['cls_name']).rsplit('.', 1)[-1]})")
    return ", ".join(out)


def _without_project(metrics, dotted):
    """A metrics dict as plain data, each cls_name relative to the config's own project."""
    prefix = dotted.split(".configs.")[0] + "."
    return {
        key: {**entry, "cls_name": str(entry["cls_name"]).removeprefix(prefix)}
        for key, entry in metrics.to_dict().items()
    }


def _render(field, value):
    if field.endswith("_metrics"):
        modules = _metric_modules(value)
        if not modules:
            return f"`{field}` is empty: built in `dict_metric_from_list()` instead"
        return f"`{field}`: {modules}"
    return f"`{field}` = `{value}`"


def _sub_losses(cfg, dotted):
    """Each term's module in a composite loss, relative to the config's own project.

    criterion_name alone hides these: four examples leave it on the builtin composite and
    put the whole point of the page in a project-local term underneath it.
    """
    params = getattr(cfg, "criterion_params", None)
    if params is None or "dict_of_loss_params" not in params:
        return {}
    prefix = dotted.split(".configs.")[0] + "."
    terms = params["dict_of_loss_params"]
    return {key: str(terms[key].get("cls_name", "")).removeprefix(prefix) for key in terms.keys()}


def _config(dotted):
    _on_path()
    return importlib.import_module(dotted).get_config()


def spine(dotted):
    """The markdown block for one example config."""
    ref, cfg, default = _config(REF), _config(dotted), base_config()
    rows = {}
    for field, step in SHAPE.items():
        value = getattr(cfg, field, None)
        ref_value = getattr(ref, field, None)
        if field.endswith("_metrics"):
            # Every project carries its own copy of a metric, so the same module under
            # another project's prefix is not a change.
            if _without_project(value, dotted) == _without_project(ref_value, REF):
                continue
        elif value == ref_value:
            continue
        # The baseline pins some fields the framework leaves alone -- amp_dtype="fp32" for
        # determinism, a warmup schedule. An example sitting on the default differs from
        # the baseline without having changed anything itself.
        if value == getattr(default, field, None):
            continue
        # Every task writes its own model, so a different model_name is not news.
        if field == "model_name":
            continue
        rows.setdefault(step, []).append(_render(field, value))

    # An added engine declares its own metrics field, which is not in SHAPE by name.
    for field in sorted(k for k in cfg.keys() if k.endswith("_metrics") and k not in SHAPE):
        if getattr(cfg, field, None) != getattr(ref, field, None):
            rows.setdefault("metrics", []).append(_render(field, getattr(cfg, field)))

    # Every project carries its own copy of a loss term, as with metrics above.
    terms, ref_terms = _sub_losses(cfg, dotted), _sub_losses(ref, REF)
    if terms != ref_terms and terms:
        shown = ", ".join(f"`{k}` ({v.rsplit('.', 1)[-1]})" for k, v in sorted(terms.items()))
        rows.setdefault("loss", []).append(f"`dict_of_loss_params`: {shown}")

    trainer = importlib.import_module(cfg.main_runner).Trainer
    overrides = sorted(n for n in vars(trainer) if not n.startswith("__") and n in BASE_METHODS)
    for name in overrides:
        if name == "prep_batch":
            continue
        rows.setdefault(STEP_OF_METHOD.get(name, "prep-batch"), []).append(f"overrides `{name}()`")

    if not rows:
        return NOTHING

    same = [TITLE[s] for s in ORDER if s not in rows]
    out = []
    if same:
        # "Nothing beyond", not "same as": every task writes its own model, dataset and
        # prep_batch. What this asserts is that none of them needs anything the walk did
        # not already cover -- not that the files are identical.
        out.append(f"**Nothing beyond [the walk](../steps/dataset.md):** {' · '.join(same)}\n")
    out += ["| step | what this example changes |", "|---|---|"]
    for step in ORDER:
        if step in rows:
            out.append(f"| [{TITLE[step]}](../steps/{step}.md) | {'<br>'.join(rows[step])} |")
    return "\n".join(out)


def on_page_markdown(markdown, page, config, files):
    def replace(match):
        example, stem = match.group(1), match.group(2)
        dotted = f"examples.{example}.configs.{stem}"
        try:
            return spine(dotted)
        except Exception as exc:
            raise ValueError(
                f"{page.file.src_uri}: spine marker '{example}/{stem}' does not resolve "
                f"to a usable config ({dotted}): {exc}"
            ) from exc

    return MARKER.sub(replace, markdown)
