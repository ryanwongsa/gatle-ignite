"""Early stopping: ignite's, plus a floor before it arms and one score every rank agrees on."""

import ignite.distributed as idist
from ignite.handlers import EarlyStopping

from gatle_ignite.dispatch import ConfigError

# Keys ignite has always written. A checkpoint missing one is damaged, not merely old.
_CARRIED = ("counter", "best_score")


def early_stop_settings(spec, cfg):
    """-> (patience, after) if `spec`'s engine may end the run, else None.

    Patience counts this engine's runs, so `every_val = 5` waits `patience x 5` epochs.
    """
    if spec.early_stop_patience_field is None or spec.score_field is None:
        return None
    patience = cfg.get(spec.early_stop_patience_field, 0) or 0
    if patience <= 0:
        return None
    if not cfg.get(spec.score_field, None):
        raise ConfigError(
            f"cfg.{spec.early_stop_patience_field} = {patience} but cfg.{spec.score_field} "
            f"is unset, so there is no metric to stop on.\n"
            f"  Set cfg.{spec.score_field}, or leave cfg.{spec.early_stop_patience_field} at 0."
        )
    after = cfg.get(spec.early_stop_after_field, 0) or 0
    return int(patience), int(after)


def build_early_stoppers(specs, engines, cfg, score_function):
    """The stoppers this config asks for, by checkpoint key. -> dict, empty if none.

    `score_function(score_name, score_factor)` builds the score each one watches.
    """
    built = {}
    for spec in specs:
        settings = early_stop_settings(spec, cfg)
        if settings is None:
            continue
        if engines.get(spec.key) is None:
            raise ConfigError(
                f"cfg.{spec.early_stop_patience_field} asks to stop on the "
                f"{spec.engine_type!r} engine, which this config never builds "
                f"(cfg.{spec.every_field} = {cfg.get(spec.every_field, 0)}, or no dataloader).\n"
                f"  Enable that engine, or leave cfg.{spec.early_stop_patience_field} at 0."
            )
        patience, after = settings
        built[f"early_stop_{spec.engine_type}"] = EarlyStop(
            spec=spec,
            patience=patience,
            after=after,
            score_function=score_function(
                cfg.get(spec.score_field), cfg.get(spec.score_factor_field, 1)
            ),
            trainer=engines["trainer"],
        )
    return built


class EarlyStop:
    """A `_post_eval` handler: called with the eval engine after `run()` returns."""

    def __init__(self, spec, patience, after, score_function, trainer):
        self.spec = spec
        self.after = after
        self.trainer = trainer
        self._score = score_function
        self._inner = EarlyStopping(
            patience=patience, score_function=self._agreed_score, trainer=trainer
        )

    def _agreed_score(self, engine):
        """Rank 0's score, on every rank, so every rank's counter stays identical.

        Broadcast in `finally`: if rank 0 raises, the others would block until the NCCL timeout.
        """
        world, value = idist.get_world_size(), 0.0
        try:
            if idist.get_rank() == 0:
                value = float(self._score(engine))
        finally:
            if world > 1:
                value = float(idist.broadcast(value, src=0))
        return value

    def __call__(self, engine):
        epoch = self.trainer.state.epoch
        if epoch < self.after:
            # Below the floor the counter does not run, so a plateau is counted from here.
            return
        before = self._inner.counter
        self._inner(engine)
        counter, patience = self._inner.counter, self._inner.patience
        # Not trainer.should_terminate: another stopper, TerminateOnNan or a user may set it.
        stopping = counter >= patience
        if idist.get_rank() != 0:
            return
        if stopping:
            print(
                f"[early-stop] {self.spec.engine_type}: no improvement for "
                f"{counter} evaluations -- stopping at epoch {epoch}"
            )
        elif counter > before:
            print(
                f"[early-stop] {self.spec.engine_type}: {counter}/{patience} "
                f"evaluations with no improvement"
            )

    def state_dict(self):
        return self._inner.state_dict()

    def load_state_dict(self, state_dict):
        # ignite adds required keys across releases: fill missing ones from the live handler.
        # Never _CARRIED: padding those would silently restart the count from a damaged file.
        state_dict = dict(state_dict)
        for key in getattr(EarlyStopping, "_state_dict_all_req_keys", ()):
            if key not in state_dict and key not in _CARRIED:
                state_dict[key] = getattr(self._inner, key, None)
        self._inner.load_state_dict(state_dict)
