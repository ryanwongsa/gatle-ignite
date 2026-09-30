# Your first task

A complete gatle-ignite task is six small files, one per concern. Only one of them knows anything
about training, and none of them contain a training loop.

```text
synthetic/
├── configs/
│   ├── base_utils.py            # where checkpoints go. Not a config.
│   └── synthetic_v0.py          # 6. get_config(): names the other five by dotted path
├── dataloaders/
│   └── synthetic_dataset.py     # 1. get_ds()
├── losses/
│   └── loss_functions/
│       └── cross_entropy.py     # 4. Loss
├── metrics/
│   └── accuracy.py              # 5. get_metric()
├── models/
│   └── mlp.py                   # 2. Model
└── trainer/
    └── synthetic_trainer.py     # 3. Trainer: prep_batch, and nothing else
```

One directory per concern, and a concern you do not have is a directory you do not create: this task
has no augmentation, so there is no `augmentation/`.

`gatle-ignite init my_project` writes this tree, plus a `README.md` saying where an augmentation or
a sampler would go and a `.gitignore` that keeps checkpoints out of git. The only differences from
what you see here are the project's name in two filenames (`my_project_v0.py`, `my_project_trainer.py`) and the
`examples.synthetic.` prefix, which an example needs to be importable from the repo root and a
project does not: a project *is* its own root.

!!! note "The config is the one file that cannot use relative imports"
    It is loaded by **file path**, so it has no parent package: `from .vocab import X` raises. Use an
    absolute import: `from dataloaders.data_utils.vocab import X`. Every other module in the same directory is
    imported by dotted path and may use relative imports normally, which is exactly why this catches
    people. You get a plain error saying so rather than a traceback.

## Now build one

Work through the nine steps in order. Each is one page: what the file must expose, a real example of
it, and the config keys that select it.

[Start at step 1: the dataset](../steps/dataset.md){ .md-button .md-button--primary }

The files above are the real [`examples/synthetic/`](../examples/synthetic.md), which the test suite trains
on every commit. The steps include them verbatim rather than retyping them. To see it run first:

```bash
gatle-ignite train --config=examples/synthetic/configs/synthetic_v0.py
```
