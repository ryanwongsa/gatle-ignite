# Diffusion (eval is not a forward pass)

A DDPM over eight Gaussians on a ring. CPU, no downloads, ~40 s.

```bash
gatle-ignite train --config=examples/diffusion/configs/diffusion_v0.py
python examples/diffusion/scripts/check_sampling.py     # the proof
```

Two things here are not standard supervised learning, and both land on existing hooks.

<!-- spine: diffusion/diffusion_v0 -->

## The target is sampled per batch, not supplied by the dataset

`prep_batch` draws a timestep and a noise vector and puts them in `targets`. Nothing in the framework
assumes targets come from the dataloader, so a plain MSE term wires up to `("targets", "noise")` like
any other label.

```python title="examples/diffusion/trainer/diffusion_trainer.py"
--8<-- "examples/diffusion/trainer/diffusion_trainer.py"
```

## Eval runs the reverse process

`eval_step` is overridden to **sample** (200 ancestral steps from pure noise) instead of doing a
forward pass. This is the normal case for generative models, retrieval and beam search, not an
exotic one.

## Why the loss is not the evidence

`train/loss_eps_avg` sits at ~0.31 for the whole run and barely moves. A noise-predictor's loss falls
even when sampling is broken, so it tells you nothing. The metrics compare *sampled* points against
the true ring.

`check_sampling.py` scores four columns, and the two controls are the point:

```text
metric                  real vs real  prior N(0,I)     UNTRAINED       TRAINED
valid/mode_dist               0.0615        0.6084    27524.3694        0.1940
```

- **UNTRAINED** diverges to ~2.7e4, proving the sampler is really driven by the weights.
- **prior N(0,I)**, a sampler that never denoises, is the sharper control, and it is why the score
  is `valid/mode_dist`. On `valid/mode_tv` pure noise scores 0.055 against real data's own 0.057
  floor: isotropic noise spreads across eight modes as evenly as the ring does, so `mode_tv` *cannot
  distinguish a working sampler from no sampler*. `mode_dist` separates them, 0.19 vs 0.61.

Honest limit: trained `mode_dist` 0.194 against a 0.062 floor. The samples sit on the right ring at
the right radius with the right mode occupancy, and are ~3x fuzzier than real data. It learned the
distribution; it did not nail it.
