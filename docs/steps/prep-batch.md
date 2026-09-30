# 3. `prep_batch`

`trainer/my_trainer.py` → `cfg.main_runner`

```python
class Trainer(BaseTrainer):
    def prep_batch(self, batch, split="train", **kwargs) -> dict
```

This is normally the **whole** task-specific trainer. Engines, metrics, checkpointing, resume,
logging, AMP and DDP are inherited.

```python title="examples/synthetic/trainer/synthetic_trainer.py"
--8<-- "examples/synthetic/trainer/synthetic_trainer.py"
```

- **`model_input`** is splatted into the model's `forward`, so its keys match that signature.
- **`targets`** is what the loss and metrics resolve their `tgt_name` paths against.

`split` is the engine's `engine_type`: `"train"`, `"valid"`, `"test"`, or whatever an added engine
calls itself. Read `split`, never `kwargs`: a mistyped `kwargs` key returns `None` and silently takes
the wrong path.

The launcher constructs `Trainer(local_rank, cfg)` and drives the lifecycle, so you never call
`fit()` yourself.

`train_step` and `eval_step` are the hooks below this one: override `eval_step` when eval is not a
forward pass (a diffusion sampler, an autoregressive decode), and `train_step` when the step itself
is not standard supervised. Both have working defaults, as do `build_model`, `build_dataloaders`,
`forward`, `backward`, `eval_specs`, `train_spec`, `eval_context` and
`extra_to_save`, which are documented in the **[Python API](../reference/api.md)**.

!!! tip "The loss receives `prep_batch`'s whole return"
    Not `x["targets"]`. Every contract in this framework passes `(y_pred, target)` where `target` is
    what you returned here, which is why `tgt_name` paths start `("targets", ...)`.

**Templates:** [trainer](../templates.md#trainer)
