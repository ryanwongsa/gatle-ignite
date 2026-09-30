# Verifying on real hardware

Some claims in [Status](../status.md) need GPUs or a second machine to check. These scripts check
them on real hardware. The test suite never runs them.

You need the repo checked out, not just the wheel. The wheel ships neither `scripts/` nor
`examples/`, and every script trains `examples/synthetic`, so `import examples...` raises
`ModuleNotFoundError` with only the wheel installed. A wheel built from one commit beside scripts
from another verifies the **wheel**, not your checkout: say which you ran.

```bash
git clone https://github.com/ryanwongsa/gatle-ignite.git
cd gatle-ignite && pip install -e .
```

Every script needs `PYTHONPATH=$(pwd)` and must run from the repo root. Each prints a
`[PASS]`/`[FAIL]` line per check with the number it observed, then a tally. Exit 0 means all passed.

## What to do with the result

| script | hardware |
|---|---|
| `verify_multigpu.py` | ≥2 GPUs, one box |
| `verify_multinode.py` | 2 machines, no GPU |
| `verify_compile.py` | ≥1 GPU (≥2 for its DDP check) |

Open an issue with:

1. **The whole output.** `8/8 passed` is not evidence; the numbers are.
2. **The hardware:** GPU model and count, driver, `torch.__version__`, ignite version. Without it
   "Verified" names no particular thing.

A **green** run moves that claim out of *Not verified*, **scoped to what actually ran**. One run on
one card is not "`nccl` works", it is "`nccl` worked here, once". A **red** run is the script
working: include the failing check and its output.

!!! danger "Never loosen a check to make it pass"
    Each check exists because the thing it checks fails *silently*: the run completes and reports
    success while being wrong. A check that cannot fail is indistinguishable from one that passes.

What each script has and has not run is recorded in
[Status](../status.md#verification-scripts).

## `verify_multigpu.py`

```bash
PYTHONPATH=$(pwd) python scripts/verify_multigpu.py
```

A couple of minutes. Checks `nccl` and exact sharding (a 2-rank run against 1-rank truth over an eval
split of 127, which deliberately does not divide), `sync_bn` with a BatchNorm model and a BN-free
control, and autocast (a `GradScaler` for fp16 and not for bf16).

## `verify_multinode.py`

Run the **same command on every machine**; `node_rank` is the only difference.

```bash
# node 0, reachable at 10.0.0.1
PYTHONPATH=$(pwd) MASTER_ADDR=10.0.0.1 python scripts/verify_multinode.py 0
# node 1
PYTHONPATH=$(pwd) MASTER_ADDR=10.0.0.1 python scripts/verify_multinode.py 1
```

`MASTER_ADDR` is required and must point at node 0; localhost is refused, because every node would
rendezvous with itself and "pass" on two separate worlds. `MASTER_PORT` defaults to `29500` and must
be open between the machines; `NNODES` defaults to `2`. No GPU needed: the claim is about
rendezvous, so two cheap CPU boxes over `gloo` are enough.

**Paste the output from both nodes.** One node's output cannot prove the run was multi-node, which is
the whole failure mode. The load-bearing check is `world_size`: a node that rendezvoused with itself
reports its *local* size and looks healthy otherwise, and exact sharding over a split world is still
exact, so the metric check reports the right number either way.

If nothing happens it is not hung: rendezvous times out after 10 minutes and names the likely causes.
That timeout output is worth reporting too.

## `verify_compile.py`

```bash
PYTHONPATH=$(pwd) python scripts/verify_compile.py
```

Checks the guard (1), checkpoints crossing the compile boundary (2 to 4), bf16 loss parity
compiled against eager (5), fp16 with a `GradScaler` (6), 2-GPU `nccl` training with compile on
(7), and `cfg.compile = "auto"` resolving **on** (8). Checks 2 to 6 build the classification
example's CNN; check 7 trains `examples/synthetic`.

On a CPU-only machine, 6 and 7 print `[SKIP]` and 8 expects **off**. A skip is never a pass, so
that run proves nothing the test suite does not.
