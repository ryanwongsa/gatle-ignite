"""Checkpointing for a BaseTrainer: what is saved, "latest" and "best", resume, and eval loading."""

import math
import warnings

import ignite.distributed as idist
import torch
from ignite.handlers import Checkpoint, DiskSaver, global_step_from_engine

from gatle_ignite.dispatch import ConfigError
from gatle_ignite.utils.batch import get_value
from gatle_ignite.utils.checkpoints import (
    LATEST_PREFIX,
    RunMeta,
    best_prefix_for,
    find_latest_checkpoint,
    load_weights,
    resolve_eval_checkpoint,
)


def _to_save(trainer):
    to_save = {
        "model": trainer.model,
        "optimizer": trainer.optimizer,
        "trainer": trainer.engines["trainer"],
        "run_meta": RunMeta(trainer.cfg.max_epochs),
    }
    # ENHANCEMENT (early stopping): empty unless configured, so no new checkpoint key.
    to_save.update(trainer._early_stoppers)
    if trainer.scaler is not None:
        to_save["scaler"] = trainer.scaler
    if trainer.scheduler is not None:
        # attach_scheduler may return a list, which has no state_dict of its own.
        schedulers = (
            trainer.scheduler if isinstance(trainer.scheduler, list) else [trainer.scheduler]
        )
        stateful = [s for s in schedulers if hasattr(s, "state_dict")]
        if len(stateful) != len(schedulers):
            warnings.warn(
                f"{len(schedulers) - len(stateful)} of {len(schedulers)} scheduler(s) have no "
                f"state_dict, so their LR state is not checkpointed and is lost on resume.\n"
                f"  Give each scheduler state_dict() and load_state_dict() to keep it.",
                stacklevel=2,
            )
        for i, sched in enumerate(stateful):
            to_save["scheduler" if len(stateful) == 1 else f"scheduler_{i}"] = sched

    # Checked against the finished dict: "scaler" and "scheduler" exist only on some runs.
    for key, obj in trainer.extra_to_save().items():
        if key in to_save:
            raise ConfigError(
                f"extra_to_save() key {key!r} collides with the framework's own {key!r}.\n"
                f"  Rename it in extra_to_save()."
            )
        to_save[key] = obj
    return to_save


def score_function(score_name, score_factor):
    def score(engine):
        return score_factor * float(get_value(engine.state.metrics, score_name))

    return score


def attach_checkpoints(trainer):
    """Build "latest" for fit() to register, and queue each scoring engine's score check."""
    cfg = trainer.cfg
    # Cleared first, so a second setup() cannot register twice and inference has none.
    trainer._post_eval = {}
    trainer._latest = None
    if "run" in cfg:
        return  # an inference run must never overwrite the checkpoint it evaluates
    engine = trainer.engines["trainer"]
    to_save = _to_save(trainer) if cfg.save_ckpt else None

    for spec in trainer.eval_specs():
        if spec.score_field is None:
            continue
        _attach_best(
            trainer,
            spec=spec,
            score_name=cfg.get(spec.score_field, None),
            score_factor=cfg.get(spec.score_factor_field, 1),
            prefix=spec.ckpt_prefix,
            to_save=to_save,
        )

    if to_save is None:
        return
    trainer._latest = Checkpoint(
        to_save,
        DiskSaver(dirname=cfg.save_dir, require_empty=False),
        n_saved=cfg.n_saved,
        global_step_transform=global_step_from_engine(engine),
        filename_prefix=LATEST_PREFIX,
    )


def _attach_best(trainer, spec, score_name, score_factor, prefix, to_save):
    """Stop the run on a non-finite score, else save "best" (when `to_save` is not None).

    The check runs even with save_ckpt off, so early stopping never counts a NaN score.
    """
    if trainer.engines.get(spec.key) is None or not score_name:
        return
    best = None
    if to_save is not None:
        best = Checkpoint(
            to_save,
            DiskSaver(dirname=trainer.cfg.save_dir, require_empty=False),
            n_saved=trainer.cfg.n_saved,
            filename_prefix=best_prefix_for(prefix),
            score_function=score_function(score_name, score_factor),
            score_name=None,  # keeps the filename ending `_<score>.pt`, which score_of() parses
            global_step_transform=global_step_from_engine(trainer.engines["trainer"]),
            greater_or_equal=True,
        )

    def check_then_save(engine):
        if trainer._diverged:
            return  # an earlier engine this epoch found the model diverged
        value = float(get_value(engine.state.metrics, score_name))
        if idist.get_world_size() > 1:
            # Summed, so every rank stops together even when the metric is rank-local.
            value = idist.all_reduce(value)
        if math.isfinite(value):
            if best is not None:
                best(engine)
            return
        # A diverged model is never saved, and its weights cannot recover, so the run ends.
        trainer_engine = trainer.engines["trainer"]
        warnings.warn(
            f"training stopped: {score_name} is {value} at epoch {trainer_engine.state.epoch}; "
            f"not saved as best\n"
            f"  The run has probably diverged: check the loss and the learning rate.",
            stacklevel=2,
        )
        trainer._diverged = True
        trainer_engine.terminate()

    # Called after run() returns, outside eval_context, so a swapped-in eval model is never saved.
    trainer._post_eval.setdefault(spec.key, []).append(check_then_save)


def load_checkpoints(trainer):
    """Load weights (model_checkpoint_dir) and/or resume a run (resume). -> loaded anything."""
    cfg = trainer.cfg
    loaded = False

    if cfg.model_checkpoint_dir:
        print("LOADING MODEL FROM:", cfg.model_checkpoint_dir)
        # The loader a task uses for a second model, so both read a checkpoint the same way.
        load_weights(cfg.model_checkpoint_dir, trainer.model, strict=cfg.strict)
        loaded = True

    if cfg.resume:
        latest = find_latest_checkpoint(cfg.save_dir)
        if latest is not None:
            print("RESUMING FROM:", latest)
            _resume_from(trainer, torch.load(latest, map_location="cpu"))
            loaded = True
        else:
            warnings.warn(
                f"resume=True but {cfg.save_dir} has no '{LATEST_PREFIX}*' checkpoint, "
                f"so this run starts from scratch.\n"
                f"  Check save_dir if you meant to resume.",
                stacklevel=2,
            )
    return loaded


def _resume_from(trainer, state):
    objects = _to_save(trainer)

    # Only the keys the checkpoint holds: load_objects rejects the whole dict over one missing.
    present = {k: v for k, v in objects.items() if k in state}
    absent = sorted(set(objects) - set(present))
    if absent:
        warnings.warn(f"checkpoint has no state for {absent}; resuming without it.", stacklevel=2)
    Checkpoint.load_objects(to_load=present, checkpoint=state)

    meta = present.get("run_meta")
    restored = getattr(meta, "restored_max_epochs", None)
    if trainer.scheduler is not None and restored not in (None, trainer.cfg.max_epochs):
        warnings.warn(
            f"resuming a run saved with max_epochs={restored}, but this config says "
            f"{trainer.cfg.max_epochs}; the LR still follows the {restored}-epoch schedule.\n"
            f"  Keep max_epochs unchanged, or start fresh from model_checkpoint_dir.",
            stacklevel=2,
        )


def load_eval_checkpoint(trainer):
    """Load the model and extra_to_save() state that `gatle-ignite eval` scores. -> its path.

    Without the extra state an EMA run would score its raw weights, not the model selected.
    """
    cfg = trainer.cfg
    ckpt = resolve_eval_checkpoint(cfg.save_dir, cfg.get("load_from_ckpt", "best"), prefix="valid_")
    if ckpt is None:
        raise ConfigError(
            f"no {cfg.get('load_from_ckpt', 'best')!r} checkpoint found in {cfg.save_dir}"
        )
    print("LOADING CHECKPOINT:", ckpt)
    state = torch.load(ckpt, map_location="cpu")
    Checkpoint.load_objects(to_load={"model": trainer.model}, checkpoint=state, strict=cfg.strict)

    # One key at a time, so a checkpoint that predates a key still loads the rest.
    for key, obj in trainer.extra_to_save().items():
        if key in state:
            Checkpoint.load_objects(to_load={key: obj}, checkpoint=state)
        else:
            warnings.warn(
                f"{ckpt} has no {key!r}; evaluating with a freshly built {key!r}.\n"
                f"  That is expected if this checkpoint predates extra_to_save().",
                stacklevel=2,
            )
    return ckpt
