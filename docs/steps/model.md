# 2. The model

`models/my_net.py` → `cfg.model_name`, `cfg.model_params`

```python
class Model(nn.Module):
    def forward(self, x, mask=None) -> dict     # parameter names are prep_batch's keys
```

`cfg.model_params` is splatted into `Model(**params)`, and `prep_batch`'s `model_input` is splatted
into `forward`, so **name the parameters after the keys you send**, and a mistyped key raises
`TypeError` instead of going quiet. **Forward returns a dict**: that is what lets a loss or a metric
select one output by name.

```python title="examples/synthetic/models/mlp.py"
--8<-- "examples/synthetic/models/mlp.py"
```

!!! tip "The dict keys are an interface"
    Whatever you return here is what [the loss](loss.md) and [metrics](metrics.md) select by name.
    Rename one and `get_value` raises `ConfigError: could not resolve 'logits'; available: ['logit']`.
    It names the keys it found, so read that list before suspecting the model.

**Templates:** [model](../templates.md#model)
