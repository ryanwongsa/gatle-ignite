# Secrets and the environment

Credentials are read from the **ambient environment only**. A module holding a secret leaks it into
every process that imports it and every git history that touches the file.

| Backend | Reads |
|---|---|
| `wandb` | `WANDB_API_KEY`; `cfg.wandb_entity` (optional) selects the entity |
| `discord` | `DISCORD_WEBHOOK_URL`, or `cfg.discord_url`. Prefer the variable: a config is a file and files get committed |

Non-secret knobs `wandb` or `torch` read themselves (`WANDB_HOST`, `WANDB_DISABLE_CODE`,
`TOKENIZERS_PARALLELISM`) go the same way. This page is about getting any of them into the process.

## The `.env` file

Keep both in one file, outside git, readable only by you:

```bash
# ~/.secrets/myproj.env   (chmod 600, and gitignore it)
WANDB_API_KEY=...
DISCORD_WEBHOOK_URL=...
WANDB_DISABLE_CODE=true
TOKENIZERS_PARALLELISM=false
```

```bash
chmod 600 ~/.secrets/myproj.env      # -rw------- : only your user can read it
```

Plain, unquoted `KEY=VALUE` with no `export`: understood by both a shell `source` and a container's
`--env-file`, so one file serves every launch path.

!!! danger "A secret file is only as private as its permissions"
    On a shared machine a world- or group-readable `.env` is readable by everyone on the box.

## Local

Load the file into your shell, then train. The values land in the process environment, never on a
command line:

```bash
set -a; source ~/.secrets/myproj.env; set +a
gatle-ignite train --config=configs/my_project_v0.py
```

`set -a` marks what `source` defines for export so it reaches the child. For the W&B key alone,
`wandb login` is equivalent.

!!! warning "Never put a secret on a command line"
    `WANDB_API_KEY=... gatle-ignite ...` and `--some-flag=$SECRET` both end up in `/proc/<pid>/cmdline`,
    which **any** user on the machine can read via `ps`. The environment of a running process
    (`/proc/<pid>/environ`) is readable only by you and root, so passing secrets through the
    environment is private; passing them as arguments is not.

## SLURM

Source the file *inside* the batch script: SLURM stores a copy of what you submit, so the script
must hold a reference and not the secret. A complete `sbatch` script is at
[9. Running it](../steps/running.md#launching-it).

`squeue` shows your job's id and resources but **not** its environment. Never pass a secret on the
`sbatch`/`srun` line (`--export=VAR=value` included: the submit command shows in `ps`); set it in
the environment and let the default `--export=ALL` carry it, and never `env`-dump into a shared log.

## Docker and Apptainer

A container is a fresh environment; how the host's variables cross into it differs by runtime.

**Docker does not inherit the host environment.** Hand it the file, or pass named variables through by
value-from-host. Never write the value on the command line:

```bash
docker run --env-file ~/.secrets/myproj.env  myimage  gatle-ignite train --config=...   # file
docker run -e WANDB_API_KEY                   myimage  ...   # pass-through: value from host env, not on the CLI
```

!!! warning "Docker's daemon runs as root"
    Anyone in the `docker` group can `docker inspect` a container and read its injected environment,
    and that group is effectively root, which is why many clusters disallow Docker. Never bake a
    secret into an image: it lives in the layers and travels with every push.

**Apptainer inherits the host environment** and runs as *your* uid with no privileged daemon, so it
keeps a bare process's privacy:

```bash
set -a; source ~/.secrets/myproj.env; set +a
apptainer exec --nv container.sif gatle-ignite train --config=...
# or be explicit: apptainer exec --env-file ~/.secrets/myproj.env --nv container.sif ...
```

As with Docker, do not put a secret in the image's `%environment` section: it ships with the `.sif`.

## Who can see a secret

| Channel | Another (non-root) user | Root / cluster admin |
|---|---|---|
| A running process's environment (`/proc/<pid>/environ`) | No | Yes |
| A command-line argument (`ps`, `/proc/<pid>/cmdline`) | **Yes** | Yes |
| The `.env` file | Only if its permissions allow | Yes |
| A value baked into an image / committed to git | **Yes**, wherever the image or repo goes | Yes |
| A `docker inspect` on a shared daemon | **Yes**, if in the `docker` group | Yes |

No scheme hides a secret from root. Keep it in the environment, `chmod 600`, off command lines
and out of images and git, and **rotate anything that was ever exposed**.

!!! note "Keep secrets out of `cfg` too"
    The W&B backend logs your **entire config** as run metadata, so a secret in a config field
    (the `cfg.discord_url` fallback, say) is uploaded with it and visible to anyone with
    project access. The variables above never touch `cfg` and are never logged.

## Already committed one?

Rotate it. A secret that was ever pushed is in git history, in every clone and every fork made
since. Deleting the line does not reach any of them.
