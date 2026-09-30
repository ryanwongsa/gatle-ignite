# MNIST

A realistic task, showing the pieces the [synthetic example](synthetic.md) leaves out: an
augmentation module, a label-smoothed loss, a separate test split with its own engine, and gradient
clipping.

```bash
pip install "gatle-ignite[examples]"   # needs torchvision
# or from a checkout:  pip install -e ".[examples]"
gatle-ignite train --config=examples/mnist/configs/mnist_v0.py
```

Downloads MNIST to `./data` on first run. Reaches ~97% in one epoch on CPU.

<!-- spine: mnist/mnist_v0 -->

## Augmentation, and who gets it

```python title="examples/mnist/augmentation/mnist_aug.py"
--8<-- "examples/mnist/augmentation/mnist_aug.py"
```

`cfg.aug_name` builds **one** transform and the framework passes it to **every** split. That is a
trap: a dataset that applies it blindly will augment the validation set and quietly depress every
metric you use to make decisions. So the dataset decides:

```python title="examples/mnist/dataloaders/mnist_dataset.py"
--8<-- "examples/mnist/dataloaders/mnist_dataset.py"
```
