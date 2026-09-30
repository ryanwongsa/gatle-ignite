# Install

```bash
pip install gatle-ignite
```

## Extras

Nothing optional is installed by default, and nothing optional is imported unless you ask for it:
training with `logger_name = ["text"]` never imports `wandb`.

| Extra | Pulls in | For |
|---|---|---|
| _(none)_ | torch, pytorch-ignite, ml_collections, numpy, tqdm | training |
| `wandb` | `wandb` | W&B logging |
| `discord` | `requests` | Discord webhook notifications |
| `examples` | `torchvision` | the [MNIST example](../examples/mnist.md) |
| `dev` | `pytest`, `pytest-cov`, `ruff` | running the test suite and the linter |
| `docs` | mkdocs-material, mkdocstrings, mkdocs-gen-files | building this site |

```bash
pip install "gatle-ignite[wandb,examples]"
```

## Requirements

- **Python** ≥ 3.9
- **torch** ≥ 2.4
- **pytorch-ignite** ≥ 0.4.13, < 0.6. Both 0.4 and 0.5 work, so this installs into an existing
  0.4.13 environment with no upgrade.

A fresh install needs none of the care below: pip takes the newest torch and the newest numpy, and
they agree. The pairings only matter when you hold a dependency back, which with torch is common.

!!! warning "Holding torch back? Hold numpy back too."
    numpy 2.0 was an ABI break. A torch built against numpy 1.x cannot use numpy 2 arrays, and the
    failure arrives late: an import-time `UserWarning`, then a `RuntimeError` on the first `.numpy()`
    call, often deep inside a metric:

    ```
    UserWarning: Failed to initialize NumPy: _ARRAY_API not found
    RuntimeError: Numpy is not available
    ```

    **pip will not catch this.** torch declares no numpy bound, so pip installs the broken pair
    without complaint, and this package's metadata cannot say "numpy < 2, but only with an older
    torch". torch 2.4.1 with numpy 2 has been checked and works. If you pin a torch built against
    numpy 1.x (a CUDA driver is the usual reason to pin), pin numpy with it:

    ```bash
    pip install "torch==<the version you need>" "numpy<2"
    ```

!!! note "`torchvision` decides which torch you get"
    torchvision pins the exact torch it was built against, so `pip install ".[examples]"` can move
    torch under you. This package cannot express that pairing without pinning torch itself, which
    would make it un-co-installable with anything wanting a different one. If you care which torch you
    have, install it first and let torchvision resolve against it, or pin both together.

A GPU is optional. The [synthetic example](../examples/synthetic.md) trains on CPU in seconds. With
0 or 1 GPU a run is a single process with no distributed backend, so the command does not change.

## From a checkout

```bash
git clone https://github.com/ryanwongsa/gatle-ignite.git
cd gatle-ignite
pip install -e ".[dev,docs]"
pytest -q
gatle-ignite train --config=examples/synthetic/configs/synthetic_v0.py
```
