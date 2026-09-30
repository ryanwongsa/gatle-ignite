# 7. Checkpoints and resume

Written into `cfg.save_dir`, driven by config alone: there is nothing to write.

| Filename | When |
|---|---|
| `latest_epoch_checkpoint_<n>.pt` | every epoch, after that epoch's evaluation |
| `valid_best_result_checkpoint_<n>_<score>.pt` | when `cfg.score_name` improves |
| `test_best_result_checkpoint_<n>_<score>.pt` | when `cfg.tester_score_name` improves |

```python
cfg.save_ckpt = True     # write them at all
cfg.n_saved   = 2        # how many of each prefix to keep
cfg.resume    = True     # continue from the newest latest_epoch* in save_dir
```

`resume` is safe to leave on: it warns rather than fails when there is nothing to resume from.

Because `latest` is written after evaluation, a resumed run carries early stopping's count and a
plateau scheduler's state exactly. An evaluation that crashes leaves no `latest` for its epoch, so
resuming redoes that epoch's training.

A run that diverges stops. The first NaN or Inf in any value of the step output's `losses` ends
training at that iteration, so that epoch is never evaluated or saved. A non-finite `score_name`
(or `tester_score_name`) ends it after that evaluation, with a warning naming the epoch: the score
is not saved as `best`, and the epoch writes no `latest`. The newest files are the last good ones,
and a run that was never finite leaves no `best`, so `gatle-ignite eval --ckpt best` says so rather
than load a diverged model.

## Loading weights only

```python
cfg.model_checkpoint_dir = "/path/to/checkpoint.pt"
cfg.strict = True
```

This loads the **primary** model (the one `cfg.model_name` names) and nothing else. For any other
model, call `load_weights` from your `build_model`:

```python
from gatle_ignite import load_weights

load_weights("teacher.pt", self.teacher)          # a path, or an already-loaded blob
```

It finds the state dict inside a full checkpoint or a bare `state_dict()`, aligns DDP's `module.`
prefix in whichever direction is needed, reports how many tensors matched, and raises when none did.

`gatle-ignite eval --ckpt best|latest` picks which one to score; see the
[CLI reference](../getting-started/cli.md).

Anything else a resumed run needs to *be* the same run (an EMA shadow, a prototype bank, a running
normaliser) is declared with `extra_to_save()`, which covers save, resume and `eval` together.

!!! tip "A 0% weight match is still a 'partial' match"
    `strict=False` lets torch load nothing, warn about nothing, and return normally, leaving a random
    init while the log says weights were loaded. The framework raises instead when nothing matched.

!!! tip "Handlers attached in `attach_runner` do not exist under `evaluate()`"
    `gatle-ignite eval` never calls it. Anything that must hold in both modes belongs in
    `build_engines()`, which both paths call.
