# 1. The dataset

`dataloaders/my_dataset.py` → `cfg.train_ds_name`, `valid_ds_name`, `test_ds_name`

```python
def get_ds(ds_params, transform=None)  ->  (dataloader, info)
```

`info["length"]` is required. `ds_params` is that split's `*_ds_params`. **Batch size lives there**,
not at top level, because splits want different ones.

```python title="examples/synthetic/dataloaders/synthetic_dataset.py"
--8<-- "examples/synthetic/dataloaders/synthetic_dataset.py"
```

`build_dataloader(dataset, ds_params, sampler=None)` wraps `idist.auto_dataloader`, except on an
`exact_sharding` split under DDP, which builds a plain `DataLoader` instead, forcing `drop_last=False`
and ignoring `shuffle`. It reads:

`bs` · `num_workers` · `shuffle` · `drop_last` · `pin_memory` · `collate_fn` · `sampler_params` ·
`exact_sharding`. Anything else is your dataset's own business. `bs` is the total across GPUs on
every split: under DDP both paths divide it, and `num_workers`, the way `idist.auto_dataloader` does.

A sampler or a collate is dotted like every other component:

```python
"sampler_params": {"cls_name": "dataloaders.data_utils.balanced_sampler", "params": {...}}
"collate_fn":     {"cls_name": "dataloaders.data_utils.pad_collate",      "params": {"pad_id": 0}}
```

naming modules that expose `get_sampler(dataset, **params)` and `get_collate_fn(**params)`.
`collate_fn` also takes a plain callable, which is what a `get_ds` usually has in hand.

`cfg.aug_name` builds **one** transform and hands it to every split's `get_ds`. Your dataset decides
who actually gets it.

!!! tip "Calling `build_dataloader` is not compulsory, but skipping it costs you"
    torch's `DistributedSampler` either pads an unevenly-divisible eval split or drops its tail.
    Five samples over two ranks: the default gives `[0,2,4]` and `[1,3,0]`, scoring sample 0
    **twice**; `drop_last=True` gives `[0,2]` and `[1,3]` and never scores sample 4. Either way every
    rank agrees on the same wrong number, so the trainer uses `ExactDistributedSampler` on valid and
    test. A `get_ds` that builds its own loader gets none of it, and is warned.

**Templates:** [dataset](../templates.md#dataset) · [sampler](../templates.md#sampler) ·
[collate](../templates.md#collate) · [augmentation](../templates.md#augmentation)
