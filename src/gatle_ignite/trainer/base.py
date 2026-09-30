"""BaseTrainer: everything a task should not have to write."""

import contextlib
import gc
import warnings
from functools import partial

import ignite.distributed as idist
import torch
from ignite.engine import Engine, Events
from ignite.handlers import TerminateOnNan
from ignite.metrics import Average

try:
    from ignite.handlers.tqdm_logger import ProgressBar  # ignite >= 0.5
except ImportError:
    from ignite.contrib.handlers import ProgressBar  # ignite 0.4.x

from gatle_ignite.config.validate import validate_config
from gatle_ignite.data.helpers import ExactDistributedSampler, get_aug, get_dataset
from gatle_ignite.dispatch import ConfigError, import_entrypoint
from gatle_ignite.enhancements.compile import check_build_model_override, compile_built_model
from gatle_ignite.enhancements.early_stop import build_early_stoppers
from gatle_ignite.schedulers import attach_scheduler, prep_scheduler
from gatle_ignite.trainer.checkpointing import (
    attach_checkpoints,
    load_checkpoints,
    load_eval_checkpoint,
    score_function,
)
from gatle_ignite.trainer.specs import EVAL_SPECS, TRAIN_SPEC
from gatle_ignite.utils.amp import autocast_device_type, resolve_amp_dtype
from gatle_ignite.utils.seed import setup_seed


class BaseTrainer:
    """The training loop, so your task does not have to contain one.

    Subclass it and implement `prep_batch`; engines, metrics, checkpointing, resume, logging,
    AMP and DDP are handled here and driven by the config. Point `cfg.main_runner` at the
    module holding it: the launcher constructs `Trainer(local_rank, cfg)` and calls `fit()`.
    """

    def __init__(self, local_rank, cfg):
        self.local_rank = local_rank
        self.cfg = cfg
        validate_config(cfg, specs=self.eval_specs())
        setup_seed(cfg.seed, cudnn_benchmark=cfg.cudnn_benchmark)
        self.dtype, self.autocast_enabled = resolve_amp_dtype(cfg.amp_dtype)
        self.device_type = autocast_device_type()
        # This rank's device, for a task that creates tensors (torch.randn in a sampler).
        self.device = idist.device()

        self.model = None
        # What build_model resolved cfg.compile to. None until it has run.
        self._compiling = None
        self.criterion = None
        self.optimizer = None
        self.scheduler = None
        self.scaler = None
        self.logger = None
        self.grad_clip_norm = None
        self.grad_clip_value = None
        self.accum_steps = 1
        self.transform = None
        self.dls = {}
        self.infos = {}
        self.engines = {}
        # engine key -> handlers run after that engine's run() returns, outside eval_context.
        self._post_eval = {}
        # The "latest" Checkpoint attach_checkpoints builds and fit() registers, or None.
        self._latest = None
        # Set when a score goes non-finite, so nothing after it in that epoch is saved.
        self._diverged = False
        # checkpoint key -> EarlyStop. setup() fills it before anything is checkpointed.
        self._early_stoppers = {}
        self._built = False

    # ---- Hooks: what a task actually writes ----

    def prep_batch(self, batch, split="train", **kwargs):
        """Map a raw batch to ``{"model_input": {...}, "targets": {...}}``.

        ``model_input`` is splatted into the model's forward; ``targets`` is what the loss and
        metrics select from by name. ``split`` is the engine's ``engine_type``: "train",
        "valid", "test", or an added engine's name. Read ``split``, not ``kwargs``, where a
        mistyped key is silently None.
        """
        raise NotImplementedError(
            f'{type(self).__name__} must implement prep_batch(batch, split="train") '
            'returning {"model_input": {...}, "targets": {...}}'
        )

    def eval_specs(self):
        """The eval engines, as data. Override to add one.

        def eval_specs(self):
            return super().eval_specs() + (EngineSpec.for_split("dictionary"),)
        """
        return EVAL_SPECS

    def train_spec(self):
        """The training engine, as data. Override to change what it reports.

        `total_loss=False` keeps only the per-component averages, for components that do not
        sum to anything (a GAN's loss_d + loss_g). `with_losses=False` drops all of them.
        """
        return TRAIN_SPEC

    @contextlib.contextmanager
    def eval_context(self, spec):
        """Model state for one engine's run. Released before anything is written.

        Swap in an EMA shadow or an SWA average: anything `spec`'s engine must see and the best
        checkpoint must NOT store. Use try/finally, or an exception mid-eval leaks the swap into
        the next epoch's training. Only engines run by `self.run_eval_engine(spec)` pass here.
        """
        yield

    def run_eval_engine(self, spec):
        """Run one eval engine inside its eval_context, then write its checkpoint outside.

        The post-eval handlers run after run() returns, because ignite fires COMPLETED from
        inside it, still inside the context.
        """
        engine = self.engines.get(spec.key)
        if engine is None or self.dls.get(spec.split) is None:
            return None
        with self.eval_context(spec):
            engine.run(self.dls[spec.split], max_epochs=1, epoch_length=spec.length(self.cfg))
        # Reached only on success: a failed eval must not select a checkpoint.
        for handler in self._post_eval.get(spec.key, ()):
            handler(engine)
        return engine.state.metrics

    def forward(self, model_input):
        return self.model(**model_input)

    def build_model(self):
        """Build the model. -> nn.Module, which the framework assigns to self.model.

        What it returns is THE model: the only one DDP-wrapped, handed to the optimizer, and
        checkpointed. A second model (a distillation teacher, a GAN discriminator) is built
        here too and held by you, not returned. Hold it in a list if `self` might ever become an
        nn.Module, or __setattr__ registers it into the optimizer and every checkpoint.
        """
        Model = import_entrypoint(self.cfg.model_name, "Model", field="model_name")
        model = Model(**(self.cfg.model_params or {}))
        # ENHANCEMENT (compile): on the unwrapped model, before the DDP wrap below.
        self._compiling = compile_built_model(model, self.cfg)
        if idist.get_world_size() == 1:
            # Not auto_model: with several GPUs visible it wraps in DataParallel instead.
            return model.to(self.device)
        return idist.auto_model(model, **dict(self.cfg.auto_model_params))

    def dict_metric_from_list(self, engine_type, spec, dict_metrics, info=None):
        """Build the metric dict for one engine.

        Default: dotted-path dispatch from the config, each module exposing
        get_metric(engine_type, info, **params). Override to build metrics imperatively.
        Keys are namespaced `{engine_type}/{key}`, so cfg.score_name reads "valid/acc".
        """
        if not spec:
            return dict_metrics
        # Any mapping: ml_collections turns an assigned dict into a ConfigDict, not a dict.
        if not hasattr(spec, "items"):
            raise ConfigError(
                f"{engine_type} metrics must be a mapping of name -> {{'cls_name', 'params'}}, "
                f"got {type(spec).__name__}.\n"
                f"  To build metrics imperatively instead, override dict_metric_from_list()."
            )

        for key, entry in spec.items():
            if "cls_name" not in entry:
                raise ConfigError(f"metric entry {key!r} is missing 'cls_name'")
            get_metric = import_entrypoint(
                entry["cls_name"], "get_metric", field=f"{engine_type}_metrics.{key}.cls_name"
            )
            dict_metrics[f"{engine_type}/{key}"] = get_metric(
                engine_type, info or {}, **entry.get("params", {})
            )
        return dict_metrics

    # ---- Steps: overridable, but the defaults suit most supervised tasks ----

    def loss_fn(self, y_pred, target, **kwargs):
        return self.criterion(y_pred, target, **kwargs)

    def backward(self, loss, step=True):
        """Backward for one batch, and the optimizer step when this batch ends a window.

        `loss` arrives ALREADY divided by the accumulation window. `step` is False for every
        batch but the last of a window: an override must honour it, or accumulation becomes a
        step on every micro-batch, which trains and is wrong.
        """
        model, optimizer = self.model, self.optimizer
        if self.scaler is not None:
            self.scaler.scale(loss).backward()
            if not step:
                return
            if self.grad_clip_value is not None or self.grad_clip_norm is not None:
                self.scaler.unscale_(optimizer)
                self._clip_grads(model)
            self.scaler.step(optimizer)
            self.scaler.update()
        else:
            loss.backward()
            if not step:
                return
            self._clip_grads(model)
            optimizer.step()

    def _clip_grads(self, model):
        if self.grad_clip_value is not None:
            torch.nn.utils.clip_grad_value_(model.parameters(), self.grad_clip_value)
        if self.grad_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), self.grad_clip_norm)

    def train_step(self, engine, batch, split="train"):
        engine.state.batch = None
        engine.state.output = None
        self.model.train()

        window, is_first, is_step = self._accum_window(engine)
        if is_first:
            self.optimizer.zero_grad(set_to_none=True)

        # no_sync must wrap the forward too: DDP decides during forward whether to all-reduce.
        with self._maybe_no_sync(is_step):
            x = self.prep_batch(batch, split=split)
            with torch.autocast(
                device_type=self.device_type, dtype=self.dtype, enabled=self.autocast_enabled
            ):
                y_pred = self.forward(x["model_input"])
                loss, dict_losses = self.loss_fn(y_pred, x, iteration=engine.state.iteration)

            # Outside autocast. By the window, not accum_steps: an epoch's last one can be shorter.
            self.backward(loss / window if window > 1 else loss, step=is_step)

        # The unscaled loss is reported, so train/loss_avg is comparable across accum_steps.
        return {"y_pred": y_pred, "target": x, "losses": {"loss": loss, **dict_losses}}

    def _accum_window(self, engine):
        """(window size, is this batch first in the window, does it end the window).

        Windows never straddle an epoch, and a short final window still steps.
        """
        accum = self.accum_steps
        if accum == 1:
            return 1, True, True

        length = engine.state.epoch_length or 1
        index = (engine.state.iteration - 1) % length  # 0-based, within this epoch
        start = (index // accum) * accum
        window = min(accum, length - start)
        return window, index == start, index == start + window - 1

    def _maybe_no_sync(self, is_step):
        if is_step or not hasattr(self.model, "no_sync"):
            return contextlib.nullcontext()
        return self.model.no_sync()

    def eval_step(self, engine, batch, split="valid"):
        engine.state.batch = None
        engine.state.output = None
        self.model.eval()
        x = self.prep_batch(batch, split=split)
        with torch.no_grad():
            with torch.autocast(
                device_type=self.device_type,
                dtype=self.dtype,
                enabled=self.autocast_enabled,
            ):
                y_pred = self.forward(x["model_input"])
        return {"y_pred": y_pred, "target": x}

    # ---- Framework below. A task never needs to touch any of it. ----

    def setup(self):
        """Build everything. Idempotent, so fit() and eval paths can both call it."""
        if self._built:
            return self
        cfg = self.cfg

        if "run" not in cfg:
            from gatle_ignite.callbacks.logging import LoggingCallback

            self.logger = LoggingCallback(cfg)
            self.logger.start()

        self.build_dataloaders()
        self.model = self.build_model()
        # ENHANCEMENT (compile): an overridden build_model owns it, so check it did.
        check_build_model_override(self.model, getattr(self, "_compiling", None), cfg)
        self.build_criterion()
        self.build_optimizer()
        self.build_engines()
        self.build_scheduler()
        self.attach_metrics()
        # ENHANCEMENT (early stopping): built before attach_checkpoints saves their counters.
        # Not in inference mode, which never trains and may build a different set of engines.
        if "run" not in cfg:
            self._early_stoppers = build_early_stoppers(
                self.eval_specs(), self.engines, cfg, score_function
            )
        attach_checkpoints(self)
        self._built = True
        return self

    def fit(self):
        cfg = self.cfg
        self.setup()
        load_checkpoints(self)
        self.attach_runner()
        if self._latest is not None:
            # After the eval runners, so "latest" holds what that epoch's evaluation changed.
            trainer = self.engines["trainer"]
            trainer.add_event_handler(Events.EPOCH_COMPLETED(every=1), _clear_state)

            def save_latest(engine):
                if not self._diverged:
                    self._latest(engine)

            trainer.add_event_handler(Events.EPOCH_COMPLETED(every=1), save_latest)
        self.attach_progress()
        try:
            self.engines["trainer"].run(
                self.dls["train"], max_epochs=cfg.max_epochs, epoch_length=cfg.train_length
            )
        except BaseException:
            # Mark the run failed, or the loggers record a crash as a clean finish.
            self.teardown(failed=True)
            raise
        self.teardown()
        return self

    def evaluate(self, load_checkpoint=True):
        """Load the configured checkpoint, run the eval engines once, return the metrics.

        Needs inference mode, so it never writes a "best" checkpoint of what it evaluates.
        load_checkpoint=False evaluates the weights in memory.
        """
        cfg = self.cfg
        if "run" not in cfg:
            raise ConfigError(
                "evaluate() needs inference mode, or checkpoint handlers could overwrite what "
                "it evaluates.\n"
                "  Use `gatle-ignite eval --config=...`, or set cfg.run = True before building "
                "the trainer."
            )
        self.setup()
        if load_checkpoint:
            load_eval_checkpoint(self)
        results = {}
        for spec in self.eval_specs():
            metrics = self.run_eval_engine(spec)
            if metrics:
                results.update(metrics)

        for name, value in results.items():
            print(f"{name}: {value}")
        return results

    def teardown(self, failed=False):
        if self.logger is not None:
            self.logger.finish(failed=failed)

    def build_dataloaders(self):
        """One dataloader per declared engine, from each spec's `*_ds_name`.

        Override to add a loader belonging to no engine (a dictionary pass, a prototype
        source). Call `super()` first, or the declared splits are never built.
        """
        cfg = self.cfg
        self.transform = get_aug(cfg.aug_name, cfg.aug_params)

        self.dls["train"], self.infos["train"] = get_dataset(
            cfg.train_ds_name, cfg.train_ds_params, self.transform, field="train_ds_name"
        )
        for spec in self.eval_specs():
            split = prefix = spec.split
            # Two specs may share a split (`ds_prefix`); build its loader once.
            if split in self.dls:
                continue
            name = cfg.get(f"{prefix}_ds_name", None)
            if not name:
                self.dls[split], self.infos[split] = None, None
                continue

            # Under DDP, DistributedSampler pads an uneven eval set and scores the duplicates.
            # A get_ds that builds its own loader ignores this key; _warn_if_eval_padded checks.
            ds_params = dict(cfg.get(f"{prefix}_ds_params", {}))
            ds_params.setdefault("exact_sharding", True)

            self.dls[split], self.infos[split] = get_dataset(
                name,
                ds_params,
                self.transform,
                field=f"{prefix}_ds_name",
            )
            self._warn_if_eval_padded(split, self.dls[split])
        return self.dls, self.infos

    @staticmethod
    def _warn_if_eval_padded(split, dataloader):
        """Warn if a loader the framework could not shard exactly scores duplicates under DDP."""
        if idist.get_world_size() <= 1 or dataloader is None:
            return
        sampler = getattr(dataloader, "sampler", None)
        if isinstance(sampler, ExactDistributedSampler):
            return
        dataset = getattr(dataloader, "dataset", None)
        try:
            remainder = len(dataset) % idist.get_world_size()
        except TypeError:
            return  # an iterable dataset has no length to check
        if remainder:
            warnings.warn(
                f"{split} split: {len(dataset)} samples do not divide across "
                f"{idist.get_world_size()} ranks, so {idist.get_world_size() - remainder} "
                f"repeated sample(s) are scored.\n"
                f"  Build this split with gatle_ignite.build_dataloader to shard it exactly.",
                stacklevel=2,
            )

    def build_criterion(self):
        cfg = self.cfg
        Loss = import_entrypoint(cfg.criterion_name, "Loss", field="criterion_name")
        self.criterion = Loss(**(cfg.criterion_params or {})).to(idist.device())
        return self.criterion

    def build_optimizer(self):
        cfg = self.cfg
        get_optimizer = import_entrypoint(
            cfg.optimizer_name, "get_optimizer", field="optimizer_name"
        )
        self.optimizer = get_optimizer(self.model, **(cfg.optimizer_params or {}))

        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"MODEL: {trainable:,} trainable parameters")

        # Only fp16 needs a GradScaler: bf16 has fp32's exponent range.
        if self.dtype == torch.float16 and torch.cuda.is_available():
            self.scaler = torch.amp.GradScaler("cuda")

        self.grad_clip_norm = cfg.grad_clip_norm
        self.grad_clip_value = cfg.grad_clip_value
        self.accum_steps = self._resolve_accum_steps()
        return self.optimizer

    def _resolve_accum_steps(self):
        accum = self.cfg.get("accum_steps", 1)
        if not isinstance(accum, int) or isinstance(accum, bool) or accum < 1:
            raise ConfigError(f"accum_steps must be an integer >= 1, got {accum!r}")
        if accum > 1 and type(self).train_step is not BaseTrainer.train_step:
            # Accumulation lives inside train_step, so an override skips it unless it copies it.
            warnings.warn(
                f"{type(self).__name__} overrides train_step, so accum_steps={accum} has no "
                f"effect unless the override accumulates too.\n"
                f"  Use _accum_window() there, and pass its `step` through to backward().",
                stacklevel=3,
            )
        return accum

    def build_engines(self):
        specs = self.eval_specs()
        self._check_specs_are_distinct(specs)

        train_spec = self.train_spec()
        self.engines[train_spec.key] = self._engine_for(train_spec)
        for spec in specs:
            # Present even when unbuilt: callers ask `engines["tester"] is None`.
            self.engines[spec.key] = self._engine_for(spec) if self._wants(spec) else None
        return self.engines

    def _engine_for(self, spec):
        return Engine(partial(getattr(self, spec.step), split=spec.engine_type))

    def _wants(self, spec):
        if self.dls.get(spec.split) is None:
            return False
        # Built only if it will run. Inference mode runs every engine, whatever its cadence.
        return spec.cadence(self.cfg) > 0 or "run" in self.cfg

    @staticmethod
    def _check_specs_are_distinct(specs):
        """Two engines sharing a key or an engine_type would silently overwrite each other."""
        for attr in ("key", "engine_type"):
            seen = [getattr(s, attr) for s in specs]
            dupes = sorted({v for v in seen if seen.count(v) > 1})
            if dupes:
                raise ConfigError(
                    f"eval_specs() has more than one engine with {attr} {dupes!r}. "
                    f"Each engine needs its own {attr}."
                )

    def build_scheduler(self):
        # The whole engines dict, so a schedule can be driven by an engine the task declared.
        scheduler, engine_name, event_name = prep_scheduler(
            self.cfg.lr_scheduler,
            self.cfg,
            self.dls["train"],
            self.optimizer,
            self.engines,
        )
        self.scheduler = attach_scheduler(scheduler, engine_name, event_name, self.engines)
        return self.scheduler

    def attach_metrics(self):
        for spec in (self.train_spec(), *self.eval_specs()):
            engine = self.engines.get(spec.key)
            if engine is None:
                continue
            self.init_metrics(
                engine,
                spec.engine_type,
                self.cfg.get(spec.metrics_field, {}),
                info=self.infos.get(spec.split),
                with_losses=spec.with_losses,
                total_loss=spec.total_loss,
            )

    def init_metrics(
        self, engine, engine_type, metrics_cfg, info=None, with_losses=True, total_loss=True
    ):
        dict_metrics = self.dict_metric_from_list(engine_type, metrics_cfg, {}, info=info)

        if with_losses:

            def loss_display(key):
                return lambda output: output["losses"][key]

            keys = ["loss"] if total_loss else []
            keys += [f"loss_{c}" for c in getattr(self.criterion, "crit_keys", [])]
            for key in keys:
                dict_metrics[f"{engine_type}/{key}_avg"] = Average(
                    output_transform=loss_display(key)
                )

        for name, metric in dict_metrics.items():
            metric.attach(engine, name)
        return dict_metrics

    def extra_to_save(self):
        """Your own state to put in the checkpoint. Values need state_dict/load_state_dict.

        For an EMA shadow, a prototype bank or a running normaliser. One override covers save,
        resume and `gatle-ignite eval`, so all three agree on what a run holds.
        """
        return {}

    def attach_runner(self):
        trainer = self.engines["trainer"]
        # Stop at the first non-finite loss: the weights cannot recover from it.
        trainer.add_event_handler(
            Events.ITERATION_COMPLETED, TerminateOnNan(output_transform=_losses_on_every_rank)
        )

        if self.logger is not None:
            self.logger.on_train_epoch_end(trainer, self.optimizer)
            self.logger.on_train_iteration(trainer, self.model)

        for spec in self.eval_specs():
            self._attach_eval_runner(trainer, spec)

        # ENHANCEMENT (early stopping). Not in attach_checkpoints, which returns early on inference.
        for stopper in self._early_stoppers.values():
            self._post_eval.setdefault(stopper.spec.key, []).append(stopper)

        if self.logger is not None:
            self.logger.on_completion(trainer)

    def _attach_eval_runner(self, trainer, spec):
        engine = self.engines.get(spec.key)
        if engine is None:
            return

        # Its own method: inlined in the caller's loop, every handler would late-bind the last spec.
        @trainer.on(Events.EPOCH_COMPLETED(every=spec.cadence(self.cfg)))
        def _run(_trainer):
            _clear_state(_trainer)
            self.run_eval_engine(spec)

        # Beside the handler that runs the engine, so no engine can run without being logged.
        if self.logger is not None:
            self.logger.on_valid_epoch_end(trainer, engine)

    def attach_progress(self):
        """Progress reporting, memory cleanup, and DDP epoch wiring."""
        cfg = self.cfg
        trainer = self.engines["trainer"]
        engines = [e for e in self.engines.values() if e is not None]

        if "pbar" in cfg.logger_name and idist.get_rank() == 0:
            for engine in engines:
                ProgressBar(persist=False).attach(engine)
        else:
            every = max((cfg.train_length or len(self.dls["train"])) // 10, 1)
            trainer.add_event_handler(
                Events.ITERATION_COMPLETED(every=every), _progress_printer("TRAIN")
            )
            for spec in self.eval_specs():
                if self.engines.get(spec.key) is not None:
                    self.engines[spec.key].add_event_handler(
                        Events.ITERATION_COMPLETED(every=spec.progress_every),
                        _progress_printer(spec.progress_label),
                    )

        for engine in engines:
            engine.add_event_handler(Events.EPOCH_STARTED, _clear_state)
            engine.add_event_handler(Events.EPOCH_COMPLETED, empty_cuda_cache)

        sampler = getattr(self.dls["train"], "sampler", None)
        if hasattr(sampler, "set_epoch"):
            # DistributedSampler needs the epoch, or it replays the same shuffle every epoch.
            trainer.add_event_handler(
                Events.EPOCH_STARTED, lambda engine: sampler.set_epoch(engine.state.epoch)
            )


def _losses_on_every_rank(output):
    # Summed, so a NaN on one rank stops every rank at once instead of hanging the all-reduce.
    losses = output.get("losses", {})
    if idist.get_world_size() == 1:
        return losses
    return idist.all_reduce(sum(float(value) for value in losses.values()))


def _clear_state(engine):
    engine.state.output = None
    engine.state.batch = None


def empty_cuda_cache(_engine=None):
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()


def _progress_printer(label):
    from datetime import datetime

    def report(engine):
        if idist.get_rank() != 0:
            return
        length = engine.state.epoch_length or 0
        # iteration % length is 0 on an epoch's last iteration, which would print "0/N".
        done = ((engine.state.iteration - 1) % length) + 1 if length else engine.state.iteration
        pct = f"{done / length:.2f}" if length else "?"
        stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{stamp}] {label} epoch {engine.state.epoch}: ({pct}) - {done}/{length}")

    return report
