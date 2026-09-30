# Templates

The [contracts table](reference/contracts.md) tells you a module must expose `get_sampler`. It does not
tell you what to write. These do.

Every file below lives in `examples/templates/`, and **every one is exercised by the test suite**, so
a template that stops working is caught before you copy it. Most are wired into a
single config and trained together; the collate fn and the logger backend have their own tests
instead, because neither fits that config: a collate reshapes the batch, and a backend is a sink
rather than a component. Copy the file, rename it, point a config field at its dotted path.

| You want | Copy | Config field | Where it goes |
|---|---|---|---|
| [A metric](#metric) | `examples/templates/metric.py` | `{train,val,tester}_metrics[<name>].cls_name` | `metrics/` |
| [A dataset](#dataset) | `examples/templates/dataset.py` | `train_ds_name` | `dataloaders/` |
| [A model](#model) | `examples/templates/model.py` | `model_name` | `models/` |
| [A loss term](#loss) | `examples/templates/loss.py` | `criterion_params.dict_of_loss_params[<key>].cls_name` | `losses/loss_functions/` |
| [An optimizer](#optimizer) | `examples/templates/optimizer.py` | `optimizer_name` | `optimizer/` |
| [An LR scheduler](#scheduler) | `examples/templates/scheduler.py` | `lr_scheduler` | `scheduler/` |
| [An augmentation](#augmentation) | `examples/templates/augmentation.py` | `aug_name` | `augmentation/` |
| [A sampler](#sampler) | `examples/templates/sampler.py` | `*_ds_params.sampler_params.cls_name` | `dataloaders/data_utils/` |
| [A collate fn](#collate) | `examples/templates/collate.py` | `*_ds_params.collate_fn.cls_name` | `dataloaders/data_utils/` |
| [A trainer](#trainer) | `examples/templates/trainer.py` | `main_runner` | `trainer/` |
| [A logger backend](#backend) | `examples/templates/backend.py` | `logger_name[<entry>]` | `callbacks/backends/` |

There is no registry to update and no decorator to add. The dotted path in the config *is* the
registration.

## Metric

The decorators are the whole point of this file, and they are the thing most worth getting right:
`@sync_all_reduce` on `compute()` is what makes the number correct under DDP. Without it the metric is
right on one GPU and quietly wrong on two.

```python title="examples/templates/metric.py"
--8<-- "examples/templates/metric.py"
```

## Dataset

Note where the transform is applied. The framework builds **one** transform and hands it to every
split; the dataset decides who gets it.

```python title="examples/templates/dataset.py"
--8<-- "examples/templates/dataset.py"
```

## Model

```python title="examples/templates/model.py"
--8<-- "examples/templates/model.py"
```

## Loss

A sub-loss returns a bare tensor. Only the composite returns `(total, {"loss_<key>": tensor})`.

```python title="examples/templates/loss.py"
--8<-- "examples/templates/loss.py"
```

## Optimizer

```python title="examples/templates/optimizer.py"
--8<-- "examples/templates/optimizer.py"
```

## Scheduler

The 3-tuple return is the contract: *what* to attach, to *which* engine, on *which* event.

```python title="examples/templates/scheduler.py"
--8<-- "examples/templates/scheduler.py"
```

## Augmentation

```python title="examples/templates/augmentation.py"
--8<-- "examples/templates/augmentation.py"
```

## Sampler

```python title="examples/templates/sampler.py"
--8<-- "examples/templates/sampler.py"
```

## Collate

```python title="examples/templates/collate.py"
--8<-- "examples/templates/collate.py"
```

## Trainer

In the normal case this is one `prep_batch` and nothing else. The rest is there to show you what the
hooks look like when you do need them. Delete what you don't.

```python title="examples/templates/trainer.py"
--8<-- "examples/templates/trainer.py"
```

## Backend

`logger_name` is the one field that is not a plain dotted path: it is a **list** of enabled sinks, so a
short name resolves to a builtin and anything else is a path to a module like this. Both kinds run at
once. Subclass the base: it supplies no-op `watch`/`finish`, and the framework calls both without
checking they exist.

```python title="examples/templates/backend.py"
--8<-- "examples/templates/backend.py"
```

See [`prep_batch`](steps/prep-batch.md) for the one hook most tasks write, and
[`eval_specs`](reference/api.md) for adding an engine.
