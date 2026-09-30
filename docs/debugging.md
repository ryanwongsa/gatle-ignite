# Debugging

## Check the config without touching a GPU

```bash
gatle-ignite config --config=configs/myproj_v0.py
```

Resolves the file, applies overrides, validates, and prints the result. This is the cheapest way to
find a broken config, before it reaches a queue and burns an allocation.

## Read the error

Dispatch errors name the config field at fault and list what the module actually exports:

```text
config error: module 'examples.synthetic.models.mlp' (named by config field 'optimizer_name')
must expose 'get_optimizer', but does not.
  Found: ['Model', 'nn']
```

```text
config error: could not import 'examples.synthetic.models.mpl' (from config field 'model_name'):
No module named 'examples.synthetic.models.mpl'
  Check the path is importable (on PYTHONPATH / installed).
```

A config error prints plainly and exits 2, with no traceback through framework internals hiding
the line you need to fix.

## Typos in field names

A config field the framework does not use is ignored, with no warning: configs legitimately carry
project values that only your own trainer reads. So a misspelt `max_epoch = 5` trains for the default
`max_epochs`. Check spelling against the [config reference](reference/config.md). `--override` is
stricter: naming a field that does not exist is an error.

## Shrink the run

```bash
gatle-ignite train --config=cfg.py --override train_length=5 --override max_epochs=1
```

`--override` takes any field that already exists (inventing one is an error) and literal-parses the
value. Capping `train_length` / `val_length` / `test_length` cuts an epoch to N iterations, which is
how you smoke-test a real config in seconds.

A dotted key reaches into a nested field, which is how you sweep without writing a config per point:

```bash
for lr in 1e-3 3e-4 1e-4; do
  gatle-ignite train --config=cfg.py --override optimizer_params.lr=$lr --override name=lr_$lr
done
```

Every level must already exist. `--override optimizer_params.lr_rate=3e-4` is an error naming what
`optimizer_params` actually holds, rather than quietly adding a field nothing reads while the run
trains at the default LR.

## Start from something that works

[`examples/synthetic/`](examples/synthetic.md) trains on CPU in seconds with no data. If it runs and
your task doesn't, the difference is in your files, not the framework.

## Common surprises

| Symptom | Cause |
|---|---|
| Best checkpoint is the *worst* model | Ignite keeps the maximum score. Set [`score_factor = -1`](steps/metrics.md#choosing-the-best-checkpoint) for loss/CER |
| Training stops early with `TerminateOnNan` or `training stopped: ... is nan` | A training loss or the score that picks `best` went NaN or Inf. The run stops and keeps the last good checkpoints; see [Checkpoints](steps/checkpoints.md). Check the LR and the loss |
| No test metrics | `cfg.every_test` defaults to **0**, which builds no tester engine at all |
| No validation | No `valid_ds_name` means no evaluator engine. Not an error |
| Eval metrics look depressed | Your dataset applies the augmentation to every split; `transform` is built once and passed to all of them. The dataset decides who gets augmented |
| Valid metric stuck at chance while train loss falls | Train and valid are being labelled differently: a dataset bug, not a model bug. See [the synthetic example](examples/synthetic.md#the-task) |
| A loss sits flat from epoch 1 | Shape mismatch broadcasting silently; see [below](#a-loss-that-sits-flat-from-epoch-1) |
| `"adamw"` doesn't resolve | Correct: there is no alias table. Use the [full dotted path](reference/builtins.md) |
| Every rank prints a different metric | The metric is rank-local; see [Metrics](steps/metrics.md) |
| LR wrong after resuming | You changed `max_epochs`; see [the known limitation](status.md#known-limitation) |
| `ModuleNotFoundError` for your own module | Run from the directory your config's dotted paths are relative to. The CWD is put on `sys.path` |
| `attempted relative import` in your config | The config is loaded by path, so it has no parent package. Use absolute imports; only the config is affected |
| Best checkpoint is from a run you deleted | Stale files in `save_dir`; `n_saved` only reaps what the current run wrote. Use a fresh dir per run |
| A second model (teacher, frozen encoder) trains, or bloats the checkpoint | It was assigned as a module attribute, so torch registered it. `model_checkpoint_dir` loads the **primary** model only; use [`load_weights`](steps/checkpoints.md) |

## A loss that sits flat from epoch 1

Nothing checks that your model's output and your target agree. torch will broadcast a `(B, 1)`
prediction against a `(B,)` target rather than complain, and the result is a loss that computes,
descends to a plateau, and learns nothing. The run completes at exit 0:

```python
# model returns (B, 1), targets are (B,)  ->  MSE broadcasts to (B, B)
{'train/loss_mse_avg': 1.003}   # every epoch, forever
```

torch emits one `UserWarning` about this and it is easy to miss in the log. If a loss sits flat from
epoch 1, print the two shapes before suspecting the model:

```python
def prep_batch(self, batch, split="train", **kwargs):
    x, y = batch
    return to_device({"model_input": {"x": x}, "targets": {"values": y.unsqueeze(1)}})
```

Classification hides this (cross-entropy *wants* `(B, C)` against `(B,)`), so it usually bites the
first time you write a regression task.

## Compile stops helping partway through a run

Dynamo compiles per input *shape* and gives up after 8 recompiles, then runs eager for the rest of
the process, saying so only in a log line. A loader padding every batch to one width compiles once.
On torch 2.11 a varying length costs one extra compile and settles; on **torch 2.4, this package's
floor, it does not**: dynamo guards the tensor's stride, so it is one recompile per distinct length.
Ragged batches on 2.4: pad to a fixed width, or leave `cfg.compile` off.
