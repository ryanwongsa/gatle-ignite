# gatle-ignite

[![License: BSD-3-Clause](https://img.shields.io/badge/license-BSD--3--Clause-green.svg)](https://github.com/ryanwongsa/gatle-ignite/blob/main/LICENSE)
[![Python](https://img.shields.io/badge/python-%E2%89%A5%203.9-blue.svg)](https://www.python.org/downloads/)
[![Code style: Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Docs](https://img.shields.io/badge/docs-online-blue.svg)](https://ryaniswong.com/gatle-ignite/)

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

Builtins have no special status: they are named in full, exactly like yours. (The one exception is
`cfg.logger_name`, a list of enabled sinks, which takes short names like `"text"`.)

## The whole task-specific trainer

```python
from gatle_ignite import BaseTrainer, to_device

class Trainer(BaseTrainer):
    def prep_batch(self, batch, split="train", **kwargs):
        x, y = batch
        return to_device({"model_input": {"x": x}, "targets": {"labels": y}})
```

That is not an excerpt.

## Install

```bash
pip install gatle-ignite
```

Requires Python ≥3.9, torch ≥2.4, pytorch-ignite ≥0.4.13 (0.4 and 0.5 both work). Extras are in
the [install guide](https://ryaniswong.com/gatle-ignite/getting-started/install/).

## Start a project

```bash
gatle-ignite init my_project && cd my_project
gatle-ignite train --config=configs/my_project_v0.py   # trains as-is, on synthetic data
```

One directory per concern (`configs/`, `dataloaders/`, `losses/`, `metrics/`, `models/`,
`trainer/`), and a concern you do not have is a directory you do not create. The generated README
says where a loss, a metric or a sampler goes, and which builtins mean you need not write one.

## Or read the examples

```bash
git clone https://github.com/ryanwongsa/gatle-ignite.git && cd gatle-ignite
pip install -e ".[dev]"
gatle-ignite train --config=examples/synthetic/configs/synthetic_v0.py   # CPU, seconds, nothing to download
```

Nine worked tasks, each the same shape as what `init` writes: a GAN's two optimizers, a diffusion
sampler at eval, seq2seq, distillation with a second model, EMA, contrastive.

## Documentation

- [Your first task](https://ryaniswong.com/gatle-ignite/getting-started/first-task/): six small files
- [Entrypoint contracts](https://ryaniswong.com/gatle-ignite/reference/contracts/): what each config field expects
- [Config fields](https://ryaniswong.com/gatle-ignite/reference/config/): every field, generated from the code
- [Status](https://ryaniswong.com/gatle-ignite/status/): what is verified, and what is not

## License

BSD-3-Clause. See [LICENSE](https://github.com/ryanwongsa/gatle-ignite/blob/main/LICENSE).
