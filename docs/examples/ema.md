# EMA (extra state in the checkpoint)

A shadow copy of the weights, updated every optimizer step, evaluated instead of the live weights,
and saved in the checkpoint. CPU, no downloads, ~11 s.

```bash
gatle-ignite train --config=examples/ema/configs/ema_v0.py
python examples/ema/scripts/verify.py          # 13 checks
```

EMA is the canonical silent failure. If the shadow is never updated, never swapped in, or never
restored, **training still converges and every number still looks plausible.** So this example proves
four links separately rather than reporting one accuracy.

<!-- spine: ema/ema_v0 -->

## Where each link hangs

| Link | Hook |
|---|---|
| update the shadow after each optimizer step | `backward(loss, step)`: `step` is True only when the optimizer actually steps |
| evaluate the shadow, not the live weights | `eval_context(spec)`: the state one engine's run sees |
| restore the live weights afterwards | the `finally` in that same context, so it happens before anything is written |
| save/restore the shadow | `extra_to_save()`, which covers save, resume **and** `gatle-ignite eval` |

```python title="examples/ema/trainer/ema_trainer.py"
--8<-- "examples/ema/trainer/ema_trainer.py"
```

## The evidence

A second engine (`raw`) shares the valid dataloader via `EngineSpec.for_split("raw", ds_prefix="valid")`
and evaluates the **live** weights on identical data, so the comparison is controlled:

```text
valid/acc  0.6562     (EMA weights)
raw/acc    0.4912     (live weights, same data)
```

Plus the mechanical checks, which matter more than the 16.5-point gap:

- the shadow differs from live (`max|shadow - live| = 1.84`) and shares no storage with it;
- the model at eval time **is** the shadow (`max|model_at_eval - shadow| = 0.0`);
- the live weights come back bit-exact afterwards (`max|after - before| = 0.0`);
- the shadow survives save/resume: a fresh trainer is `7.51` away from the saved shadow before
  loading and `0.0` after.

Every one of these is mutation-tested. "The shadow differs from live" alone is **blind**: it
also passes when the EMA is never updated, because a stale copy of the initial weights differs too.
So check 1 also asserts the update count and that the shadow moved from init.

!!! note "Why these two hooks exist, and not a pair of handlers"
    Both links in the middle of that table are places a hand-rolled EMA goes wrong silently. Swap the
    weights with your own handler and `gatle-ignite eval` restores only `"model"`, so it reports
    raw-weight numbers labelled `valid/acc`, and nothing says otherwise. Release the swap too late and
    the best checkpoint, written from the eval engine's `EPOCH_COMPLETED`, stores the *shadow* as
    `"model"` and loses the live weights. `eval_context` is scoped to one engine's run and released
    before anything is written, and `extra_to_save()` is read by save, resume and eval alike, so
    neither failure is reachable from this example.
