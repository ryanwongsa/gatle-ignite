# Synthetic (CPU, no download)

The smallest complete task. Random data, a two-layer MLP, no downloads, no GPU. It trains in seconds
and exists so that "does the framework work here?" is a question you can answer immediately.

```bash
gatle-ignite train --config=examples/synthetic/configs/synthetic_v0.py
```

The test suite trains this on every commit, so these files are always working code. They are the ones the walk
uses: the dataset at [step 1](../steps/dataset.md), the model at [step 2](../steps/model.md), the
trainer at [step 3](../steps/prep-batch.md).

## The task

Labels are a fixed linear function of the inputs. That is learnable, so accuracy climbing to ~0.93 is real
evidence the loop works, not noise.

In `synthetic_dataset.py`, the separate generator for the weight matrix is the interesting line. Drawing it from the same
generator as the inputs would make it depend on *how many* inputs were drawn first, so train (n=2048)
and valid (n=512) would get different labelling functions, and validation accuracy would sit at chance
forever while training loss fell. It looks like a broken model; it is a broken dataset.

## Config

```python title="examples/synthetic/configs/synthetic_v0.py"
--8<-- "examples/synthetic/configs/synthetic_v0.py"
```

`amp_dtype = "fp32"` is pinned so the example is deterministic across machines rather than picking
bf16 wherever it happens to be available.

## Expected output

```text
{'train/loss_avg': 2.312, 'train/loss_ce_avg': 2.312}
{'valid/acc': 0.223}
...
{'train/loss_avg': 0.587, 'train/loss_ce_avg': 0.587}
{'valid/acc': 0.926}
TRAINING COMPLETE after 15 epochs
```
