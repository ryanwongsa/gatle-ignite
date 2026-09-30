# GAN (two optimizers)

A non-saturating GAN on a correlated 2-D Gaussian. CPU, no downloads, ~12 s.

```bash
gatle-ignite train --config=examples/gan/configs/gan_v0.py
python examples/gan/scripts/verify.py          # the proof
```

This example exists to answer one question: **can a framework with a single `optimizer_name` field
train a GAN?** It can, and it needs no framework change, because `optimizer_name` never meant "pick a
torch optimizer". It means "name a module that returns the thing I will hold".

<!-- spine: gan/gan_v0 -->

## The two-optimizer trick

`get_optimizer` returns one object holding two real optimizers. The framework only ever asks it for
`param_groups`, `state_dict`, `load_state_dict`, so anything with those works, and it need not
subclass `torch.optim.Optimizer`.

```python title="examples/gan/optimizer/dual_optimizer.py"
--8<-- "examples/gan/optimizer/dual_optimizer.py"
```

The alternating update is an ordinary `train_step` override:

```python title="examples/gan/trainer/gan_trainer.py"
--8<-- "examples/gan/trainer/gan_trainer.py"
```

## Why the loss is not the evidence

A GAN's total loss can fall because either adversary is winning. It is not a progress signal, and
this example's `train/loss_avg` should be ignored. The evidence is a **sliced-Wasserstein distance**
between generated and real samples, which is what selects the best checkpoint (lower is better, hence
`score_factor = -1`).

`verify.py` scores three numbers through the framework's own inference mode:

```text
FLOOR     real vs real (sampling noise)        swd = 0.0387
UNTRAINED generator at init                    swd = 1.6609
TRAINED   generator, best checkpoint           swd = 0.1087
improvement over untrained: 15.3x   (required: >5.0x)
```

The **floor** is the number that makes this honest: real-vs-real scores 0.0387 rather than 0, so a
trained score of 0.1087 is "close to the data", not "suspiciously perfect". The **untrained** control
is what a no-op training loop would score, and `verify.py` exits non-zero against it, so the check
can fail, which is the only reason to trust it passing.
