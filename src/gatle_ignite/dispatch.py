"""How a config field becomes code: import the dotted path it names, take a fixed name."""

import importlib

# (label, entrypoint, contract, fields). The docs render it; `gatle-ignite config` checks `fields`.
CONTRACTS = (
    (
        "main_runner",
        "Trainer",
        "A BaseTrainer subclass. Usually overrides only prep_batch.",
        ("main_runner",),
    ),
    (
        "model_name + model_params",
        "Model",
        "Model(**model_params). Its forward returns a dict.",
        ("model_name",),
    ),
    (
        "train_ds_name / valid_ds_name / test_ds_name + *_ds_params",
        "get_ds",
        "get_ds(ds_params, transform) -> (dataloader, info). info['length'] is required. "
        "An engine declared via eval_specs() reads its own <name>_ds_name the same way.",
        ("train_ds_name", "valid_ds_name", "test_ds_name"),
    ),
    (
        "aug_name + aug_params",
        "Transformation",
        "Transformation(**aug_params), any callable. Built once and passed to every split.",
        ("aug_name",),
    ),
    (
        "criterion_name + criterion_params",
        "Loss",
        "An nn.Module Loss(**criterion_params) whose forward(y_pred, target, "
        "iteration=None) returns (total, {'loss_<key>': tensor}), plus a crit_keys "
        "attribute naming the components. Sub-losses inside a composite have a "
        "different, simpler contract: they return a bare tensor.",
        ("criterion_name",),
    ),
    (
        "optimizer_name + optimizer_params",
        "get_optimizer",
        "get_optimizer(model, **optimizer_params) -> the object the trainer holds as its "
        "optimizer. It does NOT have to subclass torch.optim.Optimizer: the framework only "
        "ever asks it for `param_groups` (LR logging), `state_dict` and `load_state_dict` "
        "(checkpointing), plus `zero_grad`/`step` if you use the default train_step. So one "
        "object holding two real optimizers is how you train a GAN: there is no second "
        "optimizer field, and none is needed. See examples/gan/optimizer/dual_optimizer.py.",
        ("optimizer_name",),
    ),
    (
        "lr_scheduler + lr_scheduler_params",
        "get_scheduler",
        "get_scheduler(cfg, train_dl, optimizer, engines) -> (scheduler, engine_name, "
        "event). `engines` is the whole dict, so engine_name may be any engine you "
        "declared, not just 'trainer'/'evaluator'.",
        ("lr_scheduler",),
    ),
    (
        "{train,val,tester}_metrics[<name>].cls_name + params",
        "get_metric",
        "get_metric(engine_type, info, **params) -> an ignite Metric. An engine declared "
        "via eval_specs() reads its own <name>_metrics.",
        (),
    ),
    (
        "*_ds_params.collate_fn.cls_name + params",
        "get_collate_fn",
        "get_collate_fn(**params) -> a callable torch passes a list of samples. Optional: "
        "collate_fn also takes a plain callable directly, which is what a get_ds usually "
        "has in hand. The dotted form is what lets a CONFIG choose one, which a "
        "variable-length task needs.",
        (),
    ),
    (
        "*_ds_params.sampler_params.cls_name + params",
        "get_sampler",
        "get_sampler(dataset, **params) -> a torch Sampler.",
        (),
    ),
    (
        "logger_name[<entry>]",
        "Backend",
        "Backend(cfg) exposing log(metrics, step=None, epoch=None), watch(model) and "
        "finish(failed=False). Subclass gatle_ignite.callbacks.logging.Backend to "
        "inherit no-op watch/finish and write only log; a standalone class must define "
        "all three, because finish() is called after every run and watch() whenever "
        "cfg.watch_grad is set. The one exception to the dotted-path rule: logger_name is "
        "a list of enabled sinks, not a component field. A short name resolves to a "
        "builtin (gatle_ignite.callbacks.backends.*); any other entry is a dotted path to "
        'your own module. "pbar" is not a backend at all: it is a progress-bar flag the '
        "trainer reads. Constructed on rank 0 only, and only if named, so an unused "
        "sink's dependency is never imported.",
        (),
    ),
)


class ConfigError(Exception):
    """A config names something that cannot be resolved, or is missing/invalid."""


def import_module_from_path(path, field=None):
    origin = f" (from config field '{field}')" if field else ""
    if not isinstance(path, str):
        raise ConfigError(f"expected a dotted module path string{origin}, got {path!r}")
    if not path:
        raise ConfigError(f"empty module path{origin}")
    try:
        return importlib.import_module(path)
    except ImportError as e:
        raise ConfigError(
            f"could not import '{path}'{origin}: {e}\n"
            f"  Check the path is importable (on PYTHONPATH / installed)."
        ) from e


def import_entrypoint(path, attr, field=None):
    """Resolve `attr` from dotted `path`, failing here rather than deep in the training loop."""
    mod = import_module_from_path(path, field=field)
    if not hasattr(mod, attr):
        public = sorted(n for n in vars(mod) if not n.startswith("_"))
        origin = f" (named by config field '{field}')" if field else ""
        raise ConfigError(
            f"module '{path}'{origin} must expose '{attr}', but does not.\n  Found: {public}"
        )
    return getattr(mod, attr)
