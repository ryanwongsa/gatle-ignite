# gatle-ignite

A config-driven [pytorch-ignite](https://pytorch-ignite.ai/) training framework.

You write your custom functions: a config, a model, a dataset, a `prep_batch`. The framework owns the
rest: engines, metrics, checkpointing, resume, logging, mixed precision, and distributed training.

## Why this exists

gatle-ignite grew out of my own work over the years, from Kaggle competitions to my PhD research and the research that followed. I like working with pytorch-ignite and built my experiments on it, and they all shared the same template on top of it: the same trainer, config layout, checkpointing and logging, which I copied into each new project and then adapted. This package gathers that template into one place, to streamline that work and make future projects easier to start: a new one begins with `gatle-ignite init` rather than another round of copying.

It is shaped the way I already wrote my experiments: a config file, a model, a dataset and a `prep_batch`, with each piece named by a plain dotted path and kept in its own folder.

gatle-ignite is an independent project, not affiliated with or endorsed by the PyTorch-Ignite team.

## The one rule

**A config module is the single source of truth, and every component is selected by a dotted module
path that gets imported at runtime.** There is no registry and no decorator: you write a module that
exposes a fixed entrypoint name, and you point a config at it.

```python
cfg.model_name     = "models.mlp"                       # a module exposing Model(**model_params)
cfg.criterion_name = "gatle_ignite.losses.composite"
cfg.optimizer_name = "gatle_ignite.optimizers.adamw"    # builtins are just modules too
```

Builtins have no special status: they are modules at real import paths, named in full. Swapping one
for your own is changing a string, and it needs no edit to this package. (The one exception is
[`cfg.logger_name`](steps/logging.md), a list of enabled
sinks, which takes short names like `"text"`.)

## The whole task-specific trainer

```python title="examples/synthetic/trainer/synthetic_trainer.py"
--8<-- "examples/synthetic/trainer/synthetic_trainer.py"
```

That is not an excerpt. `prep_batch` maps a raw batch into the two-part structure everything else
selects from by name, and there is nothing else to write.

## Where to go

- **[Install](getting-started/install.md)**: `pip install gatle-ignite`, and the optional extras.
- **[Your first task](getting-started/first-task.md)**: six small files, running on CPU in seconds.
- **[Entrypoint contracts](reference/contracts.md)**: what each config field expects a module to expose.
- **[Config fields](reference/config.md)**: every field, generated from the code.
- **[Status](status.md)**: what is verified, what is not, and the known limitation.
