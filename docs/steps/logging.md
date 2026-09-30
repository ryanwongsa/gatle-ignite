# 8. Logging

```python
cfg.logger_name = ["text", "pbar"]
cfg.log_every   = 100      # iterations between LR / grad-norm points
cfg.watch_grad  = False    # log gradient norms; costs a pass over the parameters
cfg.project_name = "gatle" # groups runs in W&B
cfg.tags = []
```

| You write | You get |
|---|---|
| `"text"` | stdout: the metrics dict each epoch, plus LR every `log_every` iterations |
| `"wandb"` | Weights & Biases |
| `"discord"` | a webhook ping |
| `"pbar"` | a progress bar: a flag, not a backend |

**This is the one exception to the dotted-path rule.** Every other component is named in full;
`logger_name` takes these short names. Every *other* entry is treated as a module path, whether or
not it contains a dot, so a bare `"mybackend"` on `sys.path` is a custom backend, and a typo'd
builtin is an import error.

## Writing one

```python
from gatle_ignite.callbacks.logging import Backend as BaseBackend

class Backend(BaseBackend):
    def __init__(self, cfg): ...
    def log(self, metrics, step=None, epoch=None): ...
```

Constructed with the whole config, not with params. Subclass `gatle_ignite.callbacks.logging.Backend` to inherit
no-op `watch(model)` and `finish(failed=False)` and write only `log`. **A standalone class must
define all three**, or a run raises `AttributeError` as it ends. `log` receives a flat
`{name: scalar}`. `wandb` and `requests` are imported **only if** you name that backend.

Credentials come from the environment: `WANDB_API_KEY`, and `DISCORD_WEBHOOK_URL`, though the
Discord backend checks `cfg.discord_url` first, and a config is a file that gets committed. See
[Secrets and the environment](../guide/environment.md).

!!! tip "Only rank 0 logs"
    Every other rank would duplicate the series. If you add a backend, do not assume your code runs
    once per process.

**Templates:** [backend](../templates.md#backend)
