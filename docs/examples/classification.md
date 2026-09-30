# Classification (the realistic one)

Synthetic 32x32 noisy shapes, five classes, class-imbalanced. CPU, no downloads, ~17 s.

```bash
gatle-ignite train --config=examples/classification/configs/classification_v0.py
python examples/classification/scripts/checks.py       # 13 checks, each falsifiable
```

[Synthetic](synthetic.md) is the smallest task that works. This is the one to crib from for a real
project: it wires up the parts an actual classification run needs: augmentation, a class-balancing
sampler, top-k and per-class metrics, warmup, and all three engines.

<!-- spine: classification/classification_v0 -->

## The point: accuracy is the wrong score

The train set is imbalanced 4:1 and the test set carries a deployment skew of 400:200:100:40:20.
Running with and without the sampler:

| | test/acc | test/bal_acc | hbar | vbar |
|---|---|---|---|---|
| sampler **on** | 0.542 | **0.688** | 0.95 | 0.85 |
| sampler **off** | **0.611** | 0.365 | **0.00** | **0.00** |

The **worse model has the higher plain accuracy**. It bought those points by abandoning two of five
classes entirely: `hbar` and `vbar` score exactly 0.00. That is why this example scores on
`valid/bal_acc`, and it is the whole reason the example exists.

Reproduce it yourself:

```bash
SHAPES_NO_SAMPLER=1 gatle-ignite train --config=examples/classification/configs/classification_v0.py
```

## Each wired-up piece can die silently

Augmentation that never fires, a sampler that never rebalances, a test engine that never runs: none
of them raise, and accuracy still looks plausible. So `checks.py` proves each one separately:

- **The sampler rebalances**: 20 000 draws give `{square .201, circle .197, ring .200, hbar .201,
  vbar .200}` against a raw distribution of `[.527, .266, .130, .052, .026]`.
- **Augmentation hits train and not eval**: the transform's call count goes `0 -> 1088` over one train
  epoch, then **stays at 1088** across a full valid+test pass.
- **The test engine ran**: it fires at epochs 5 and 10 only, and writes a `test_best_result_*`
  checkpoint.

The checks are mutation-tested. Removing the `split == "train"` guard turns the augmentation checks
red; disabling the sampler turns the balance checks red. A check nobody has watched fail is
indistinguishable from decoration.

The sampler itself is an ordinary dotted component: `sampler_params.cls_name` in
`train_ds_params`, exposing `get_sampler`:

```python title="examples/classification/dataloaders/data_utils/balanced_sampler.py"
--8<-- "examples/classification/dataloaders/data_utils/balanced_sampler.py"
```
