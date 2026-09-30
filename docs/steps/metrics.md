# 5. Metrics and scoring

`metrics/my_metric.py` → `cfg.{train,val,tester}_metrics`

```python
def get_metric(engine_type, info, **params) -> ignite.metrics.Metric
```

```python
cfg.val_metrics = {
    "acc": {"cls_name": "metrics.accuracy",
            "params": {"src_name": "logits", "tgt_name": ("targets", "labels")}},
}
```

Keys are namespaced `{engine_type}/{key}`, so this one reads `valid/acc`.

```python
cfg.every_val  = 1    # how often the validator runs, in epochs
cfg.every_test = 0    # 0, the default, builds no tester at all
```

An engine is built only if it will actually run, so `every_test = 0` builds no tester during
training. `gatle-ignite eval` builds it anyway, since evaluating is the point of that run.
A test split has its own set of the three fields on this page: `tester_metrics`, `tester_score_name`
and `tester_score_factor`.

There are no built-in metrics, because a metric describes your task. For accuracy, copy the
synthetic example's:

```python title="examples/synthetic/metrics/accuracy.py"
--8<-- "examples/synthetic/metrics/accuracy.py"
```

## Choosing the best checkpoint

```python
cfg.score_name = "valid/acc"
cfg.score_factor = 1
```

```python title="examples/classification/metrics/topk.py"
--8<-- "examples/classification/metrics/topk.py"
```

`info` carries at least `length`, and `output_transform` selects what `update()` sees.

!!! tip "Rank-local metrics lie under DDP"
    A metric accumulating into a plain Python float computes a **per-rank** result, and each rank
    silently reports only its own shard: right on one GPU, wrong on many. Accumulate into tensors
    and decorate `compute()` with `@sync_all_reduce(...)` and `reset()`/`update()` with
    `@reinit__is_reduced`, as `examples/translation/metrics/edit_distance.py` does. A metric that
    is **not** a sum (an AUC, a distributional distance) gathers instead: `idist.all_gather` in `compute()` and no
    `sync_all_reduce`, which would double-count on top of it.

!!! tip "`score_factor = -1` for anything where lower is better"
    Checkpointing keeps the **maximum**, so a loss, a CER or a WER needs `-1`. With `+1` the run
    happily saves its worst model and reports success.

Score edit distance on **decoded** output, not teacher-forced logits: a model can teacher-force well
and decode into garbage, and the logits throw away the only signal that would have noticed.

An engine you add yourself brings its own pair of fields, named from its key: `EngineSpec.for_split("raw", ...)`
returned from `eval_specs()` reads `cfg.raw_metrics` and `cfg.every_raw`, and reports under `raw/`.
A declared engine must set its cadence field or the run raises rather than silently skipping it.

If the config form cannot express what you need, override `dict_metric_from_list` and build metrics
imperatively. **Call `super()`**, or every config-declared metric is silently discarded; if
`score_name` named one of them, checkpointing then raises rather than scoring.

**Templates:** [metric](../templates.md#metric)
