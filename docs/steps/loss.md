# 4. The loss

`losses/loss_functions/my_loss.py` → a term inside `cfg.criterion_params`

`criterion_name` is required. Even a single-term loss goes through
`gatle_ignite.losses.composite`, a weighted sum built from config:

```python
cfg.criterion_name = "gatle_ignite.losses.composite"
cfg.criterion_params = {"dict_of_loss_params": {
    "ce":  {"cls_name": "losses.loss_functions.cross_entropy",
            "loss_params": {"src_name": "logits", "tgt_name": ("targets", "labels")},
            "weight": 1.0},
    "aux": {"cls_name": "losses.loss_functions.aux", "loss_params": {...},
            "weight": 0.1, "start_iteration": 1000},
}}
```

`src_name` resolves against the model's output dict; `tgt_name` against **`prep_batch`'s whole
return**, which is why the paths start `("targets", ...)`. A string is a top-level key of that
return, a tuple is a path into it.
`start_iteration` / `end_iteration` switch a term on and off as training progresses.

There are no built-in loss terms, because a loss term describes your task. For cross-entropy, copy
the synthetic example's:

```python title="examples/synthetic/losses/loss_functions/cross_entropy.py"
--8<-- "examples/synthetic/losses/loss_functions/cross_entropy.py"
```

## Writing one

```python
class Loss(nn.Module):
    def forward(self, y_pred, target, iteration=None) -> Tensor
```

```python title="examples/contrastive/losses/loss_functions/ntxent.py"
--8<-- "examples/contrastive/losses/loss_functions/ntxent.py"
```

!!! tip "A sub-loss returns a bare tensor; a whole criterion does not"
    Replacing `criterion_name` itself means returning `(total, {"loss_<key>": tensor})` and carrying a
    `crit_keys` attribute. Inside the composite, just return the scalar.

!!! tip "The weight scales the term as it is logged, not only as it is summed"
    So `loss_aux` charts its real contribution, but two terms with different weights are on
    different scales and are not comparable by eye. A gated-off term logs as exactly `0.0`.

!!! tip "CTC: log-softmax in float32, and `CTCLoss` with autocast off"
    Take `log_softmax` of `logits.float()` and call `nn.CTCLoss` inside
    `torch.autocast(..., enabled=False)`: cuDNN's CTC kernel can produce NaN under bf16.

**Templates:** [loss](../templates.md#loss)
