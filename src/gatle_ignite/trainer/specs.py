"""The engines a trainer runs, as a table: one `EngineSpec` row per engine."""

from dataclasses import dataclass
from typing import Optional

from gatle_ignite.dispatch import ConfigError


@dataclass(frozen=True)
class EngineSpec:
    """One engine, as the names of the config fields it reads."""

    key: str  # self.engines[key]
    engine_type: str  # metric namespace, and the `split` prep_batch is given
    metrics_field: str
    every_field: Optional[str] = None  # None -> always built (the trainer)
    length_field: Optional[str] = None
    score_field: Optional[str] = None  # None -> never scores a checkpoint
    score_factor_field: Optional[str] = None
    # None -> never stops the run. The tester's stay None, so training never selects on test.
    early_stop_patience_field: Optional[str] = None
    early_stop_after_field: Optional[str] = None
    ds_prefix: Optional[str] = None  # None -> engine_type. Set it to share another's loader.
    step: str = "eval_step"  # name of the method this engine drives
    # Attach loss averages at all. False for eval engines, whose step computes no loss.
    with_losses: bool = False
    # Attach the total "loss" average. Off for a GAN, whose loss_d + loss_g falls when either wins.
    total_loss: bool = True
    label: Optional[str] = None  # progress label. None -> engine_type.upper()
    progress_every: int = 100

    @property
    def split(self):
        """The dls/infos key, and the `*_ds_name` / `*_ds_params` field prefix."""
        return self.ds_prefix or self.engine_type

    @property
    def progress_label(self):
        return self.label or self.engine_type.upper()

    @property
    def ckpt_prefix(self):
        return f"{self.engine_type}_"

    @classmethod
    def for_split(cls, name, **overrides):
        """An engine whose config fields follow the convention: `every_<name>`,
        `<name>_metrics`, `<name>_length` and so on. The two built-ins spell theirs out.
        """
        return cls(
            key=name,
            engine_type=name,
            metrics_field=f"{name}_metrics",
            every_field=f"every_{name}",
            length_field=f"{name}_length",
            score_field=f"{name}_score_name",
            score_factor_field=f"{name}_score_factor",
            early_stop_patience_field=f"{name}_early_stop_patience",
            early_stop_after_field=f"{name}_early_stop_after",
            **overrides,
        )

    def cadence(self, cfg):
        """How often this engine runs, in epochs. 0 disables it."""
        if self.every_field is None:
            return 1
        value = cfg.get(self.every_field, None)
        if value is None:
            # Only a new engine gets here, and it must not silently never run.
            raise ConfigError(
                f"engine {self.key!r} declares cadence field {self.every_field!r}, which "
                f"this config does not set. Add cfg.{self.every_field} = N (0 disables it)."
            )
        return value

    def length(self, cfg):
        return cfg.get(self.length_field, None) if self.length_field else None


TRAIN_SPEC = EngineSpec(
    key="trainer",
    engine_type="train",
    metrics_field="train_metrics",
    length_field="train_length",
    step="train_step",
    with_losses=True,
)

# The two built-ins. Every irregular config-field name lives here: val_ vs valid, tester_ vs test.
EVAL_SPECS = (
    EngineSpec(
        key="evaluator",
        engine_type="valid",
        metrics_field="val_metrics",
        every_field="every_val",
        length_field="val_length",
        score_field="score_name",
        score_factor_field="score_factor",
        early_stop_patience_field="early_stop_patience",
        early_stop_after_field="early_stop_after",
    ),
    EngineSpec(
        key="tester",
        engine_type="test",
        metrics_field="tester_metrics",
        every_field="every_test",
        length_field="test_length",
        score_field="tester_score_name",
        score_factor_field="tester_score_factor",
    ),
)
