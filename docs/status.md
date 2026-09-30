# Status

`gatle-ignite` is **0.4.0**. This page says what has been run and what has not, because "it should
work" is a weaker claim than "it was run".

## Verified

Checked on CPU, on every commit.

- **Training** end to end: the synthetic example drives loss 2.31 → 0.59 and accuracy 0.22 → 0.93.
- **Checkpointing:** `latest` every epoch, `valid_best_result` on each new best, and best-selection
  by score when `n_saved > 1`.
- **Stopping a diverged run:** a NaN loss mid-epoch or a NaN `score_name` stops the run and leaves no
  `best` or `latest` from that epoch. That one rank's NaN stops every rank was run once, on a 2-rank
  gloo run, and is not in the suite.
- **Resume** restarts at the saved epoch, not from scratch.
- **Inference mode** loads a checkpoint and evaluates without training, and writes no checkpoints.
- **Metric dispatch**, composite-loss weighting and iteration gating, and `get_value` path resolution.
- **LR schedule shapes:** `warmup_epochs = N` gives an N-epoch warmup, and the other
  `lr_scheduler_params` reach the scheduler.
- **Config validation:** required fields, field values, overrides.
- **Optional deps stay optional:** a text-only run imports neither `wandb` nor `requests`.
- **Distributed:** two-process `gloo` runs shard the train loader, all-reduce metrics, and on an eval
  set that does not divide evenly across ranks report the single-process answer. Eval splits are
  sharded exactly ([`ExactDistributedSampler`](reference/api.md)), so no sample is scored twice.
- **Packaging:** the wheel, installed into a clean environment, trains a project from a directory
  outside the repo with nothing else on `sys.path`.
- **Scaffolding:** the project `gatle-ignite init` writes trains, both from the source tree and from
  the installed wheel outside the repo, which proves the template ships in the wheel.
- **The examples run:** every example except `mnist` (it downloads) and `teacher_student` (it trains
  a teacher first) trains for one short epoch. That shows it runs, not that it learns: the scripts in
  an example's `scripts/` check that, and are too slow to run on every commit.
- **`cfg.compile` on CPU:** the classification example trains through the CLI with `compile=True`, a
  compiled model's checkpoint round-trips with an eager one, both refusals fire, and a 2-rank gloo run
  compiles and trains. No speed claim is made.

## Verified once, on hardware

Weaker than the section above: each is one run, on one machine, on one day, and nothing re-runs it.
It shows the code worked then, not that it still does.

- **Multi-GPU:** 12/12 checks on 2x Tesla T4, torch 2.10.0+cu128, via `scripts/verify_multigpu.py`
  (`nccl`, exact eval sharding, `sync_bn`, bf16/fp16 autocast).

## Not verified

- **Multi-node** (`nnodes > 1`): no test has seen two nodes rendezvous. The topology maths is tested,
  and a run that would split into separate worlds is refused. Treat the first real run as the test.
- **Multi-config launch** (`--config=a.py,b.py`): the fan-out is tested, but not that each port it
  hands out is free. `pick_port` never binds to check, so a clash with another process is possible.
- **`cfg.compile` under DDP on GPUs.** A 2-rank gloo test compiles and trains on CPU, but never
  reaches NCCL or DDPOptimizer's bucketing.
- **`cfg.compile` on a GPU:** `scripts/verify_compile.py` has not been run on one.
- **External links** in these docs are not checked.

## Verification scripts

What the scripts in [Verifying on real hardware](guide/verification.md) have and have not run.

- **`verify_multigpu.py`:** only check 1 (`nccl` and exact sharding) has been seen to fail when what
  it checks is broken. Checks 2 (`sync_bn`) and 3 (autocast) need a GPU and have only run in the
  12/12 pass above. If one fails, suspect the script before the framework.
- **`verify_compile.py`:** checks 6 and 7 skip, rather than pass, when the hardware is missing.
- **A wheel beside a checkout verifies the wheel.** If they are different commits, the scripts test
  the installed code. Say which you ran; `pip install -e .` from the repo root avoids the question.

## Known limitation

**Resuming with a changed `max_epochs` restores a stale LR schedule.** A scheduler's `state_dict`
carries its geometry (a cosine's `T_max`) as well as its position, so loading it overrides what the
config just computed. Train for 3 epochs, resume with `max_epochs = 6`, and the LR keeps following
the 3-epoch curve, with nothing to say so. The framework records `max_epochs` in each checkpoint and
**warns** when a resume disagrees with it.

To extend a run, don't resume. Start fresh from the weights:

```python
cfg.resume = False
cfg.model_checkpoint_dir = "./checkpoints/my_run/latest_epoch_checkpoint_3.pt"
cfg.max_epochs = 6            # a real 6-epoch schedule, from a warm start
```
