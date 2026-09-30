# 9. Running it

Everything so far runs on a laptop unchanged. These are the knobs you reach for when it goes
somewhere real, in the order you meet them. Every field, with its type and default, is in
[Config fields](../reference/config.md).

## Where it runs

The launcher reads `torch.cuda.device_count()`, so **CPU, one GPU and N GPUs are the same config**.
You just run it elsewhere. Set the process count only to override that probe:

```python
cfg.nproc_per_node  = 8            # processes per machine; None = one per visible GPU
```

`CUDA_VISIBLE_DEVICES=0` restricts a run to one card without touching the config.

## How long, and how hard

```python
cfg.max_epochs      = 15      # default 1
cfg.grad_clip_norm  = 1.0     # and/or cfg.grad_clip_value; both default to None
```

Clipping happens **inside** `BaseTrainer.backward()`, on the batch that closes an accumulation
window, and under AMP after the scaler is unscaled, so the threshold is in real gradient units.
Setting both is not an error: value is clipped first, then norm. An override of `backward` that
never calls `super()` loses clipping silently. `examples/ema/` overrides it and calls `super()`
first, which is why it keeps it.

A run can also end before `max_epochs`, when the metric that [selects your best
checkpoint](metrics.md) has stopped improving:

```python
cfg.early_stop_patience = 50   # evaluations with no improvement. 0, the default, never stops
cfg.early_stop_after    = 500  # epochs before the counter arms. 0 arms immediately
```

Patience counts **evaluations, not epochs**: with `every_val = 5` that is `patience x 5` epochs. It
needs `score_name` and raises without one; the tester can never stop a run. The count is
checkpointed, so a resume continues it.

## Precision

```python
cfg.amp_dtype = "auto"     # auto | bf16 | fp16 | fp32
```

Leave it on `auto`: bf16 where the hardware supports it, else fp32. Pin a dtype only to reproduce
an older run exactly, since numerics differ between them.

## Speed

```python
cfg.compile        = "auto"                        # or True / False
cfg.compile_params = {"mode": "max-autotune"}      # torch's vocabulary, unvalidated here
```

`"auto"` asks whether this run *can* compile, never whether it should, so it is deterministic. The
first step compiles the forward and the second the backward, so a short run can spend more time
compiling than it saves.

## Memory

```python
cfg.accum_steps      = 4
cfg.train_ds_params  = {"bs": 16}
```

Trains as if the batch were 64, on the memory of 16. The framework divides the loss by the window
and skips DDP's all-reduce on batches that will not step. Override `backward` and you must honour
its `step` argument.

## Several machines

```python
cfg.nnodes      = 2
cfg.node_rank   = 0              # 1 on the second machine: the only line that differs
cfg.master_addr = "10.0.0.1"     # node 0, reachable from every node
cfg.master_port = 29500
```

Run the **same command** on every machine. `master_addr` left at `127.0.0.1` is refused: every node
would rendezvous with itself and train a separate model while reporting success.

## The `.env` a run sources

A W&B key or a webhook URL is not a config field. It is environment, and it must stay out of git.
Keep them in one file outside the repo, readable only by you:

```bash
# ~/.secrets/myproj.env   (chmod 600, and never inside the repo)
WANDB_API_KEY=...
DISCORD_WEBHOOK_URL=...
TOKENIZERS_PARALLELISM=false
```

```bash
chmod 600 ~/.secrets/myproj.env      # -rw------- : only your user can read it
set -a; source ~/.secrets/myproj.env; set +a
```

Plain `KEY=VALUE`, unquoted and with no `export`: that form is understood by both a shell `source`
and a container's `--env-file`, so one file serves every launch path. `set -a` exports everything
sourced between it and `set +a`, which is what `gatle_ignite` reads.

**Source it, never pass it.** A command-line argument is visible in `ps` to every other user on the
box, and SLURM stores a copy of what you submit, so a secret on the `sbatch` line is a secret you
have published. A running process's environment is not readable by other non-root users; its command
line is.

## Launching it

On your own machine, those two lines are the whole of it: source, then train:

```bash
set -a; source ~/.secrets/myproj.env; set +a
gatle-ignite train --config=configs/my_project_v0.py
```

Under a scheduler the same two lines open a batch script, and the config does not change:

```bash
#!/bin/bash
#SBATCH --job-name=myproj
#SBATCH --gres=gpu:2

set -a; source ~/.secrets/myproj.env; set +a       # secrets: sourced, never on the sbatch line
export WANDB_HOST="${SLURM_ARRAY_JOB_ID:-$(hostname)}-${SLURM_ARRAY_TASK_ID:-0}"

python -m gatle_ignite train --config=configs/my_project_v0.py
```

`python -m gatle_ignite` is identical to `gatle-ignite` and is the form to prefer under a scheduler.
Several configs at once: `--config=a.py,b.py`, each its own subprocess and port.

That is the whole of what a run needs. [Secrets and the environment](../guide/environment.md) goes
further: Docker and Apptainer, and who can see a secret through which channel.
