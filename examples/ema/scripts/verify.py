"""Proof that each EMA link works: every check fails if its link is cut."""

import copy
import shutil
from contextlib import contextmanager
from pathlib import Path

import torch
from ignite.engine import Events

from examples.ema.configs.base_utils import ckpt_dir
from examples.ema.configs.ema_v0 import get_config
from examples.ema.trainer.ema_trainer import Trainer
from gatle_ignite.trainer.checkpointing import load_checkpoints

PASS, FAIL = "PASS", "FAIL"
results = []


def check(name, ok, detail):
    results.append((name, ok, detail))
    print(f"[{PASS if ok else FAIL}] {name}\n       {detail}")


def summarise():
    print("\n" + "=" * 62)
    failed = [n for n, ok, _ in results if not ok]
    print(f"{len(results) - len(failed)}/{len(results)} checks passed")
    if failed:
        print("FAILED: " + ", ".join(failed))
        raise SystemExit(1)
    print("all EMA links verified")


def max_abs_diff(a, b):
    """Largest elementwise gap between two state dicts, over float tensors."""
    return max(
        (a[k].float() - b[k].float()).abs().max().item() for k in a if a[k].is_floating_point()
    )


def main():
    torch.manual_seed(0)
    cfg = get_config()
    cfg.max_epochs = 2
    cfg.train_length = 30  # a short run; we are testing mechanism, not accuracy

    # Its own save_dir, wiped first: `--ckpt best` picks the highest score in the directory,
    # so a leftover checkpoint from an earlier run would be preferred over this run's.
    cfg.save_dir = ckpt_dir("verify")
    shutil.rmtree(cfg.save_dir, ignore_errors=True)

    task = Trainer(0, cfg)
    task.setup()  # idempotent; fit() calls it again. Lets us snapshot the shadow at init.
    shadow_at_init = {k: v.clone() for k, v in task.ema.shadow.items()}
    task.fit()

    live = task.model.state_dict()
    shadow = task.ema.shadow

    # ---- LINK 1: the shadow differs from the live weights, and is not an alias ----

    diff = max_abs_diff(shadow, live)
    check(
        "1a. EMA shadow differs from live weights",
        diff > 1e-6,
        f"max|shadow - live| = {diff:.6f} over {len(shadow)} tensors "
        f"(ema.num_updates = {task.ema.num_updates})",
    )

    # 1a passes for an EMA never updated (it keeps the init), so check it moved once per step.
    expected_steps = cfg.max_epochs * cfg.train_length // cfg.accum_steps
    moved_from_init = max_abs_diff(shadow, shadow_at_init)
    check(
        "1c. the shadow was updated once per optimizer step",
        task.ema.num_updates == expected_steps and moved_from_init > 1e-6,
        f"ema.num_updates = {task.ema.num_updates}, expected {expected_steps} "
        f"({cfg.max_epochs} epochs x {cfg.train_length} iters / accum {cfg.accum_steps}); "
        f"max|shadow - shadow_at_init| = {moved_from_init:.6f}",
    )

    # Value comparisons cannot see aliasing, so poke the live tensor and watch the shadow stay.
    key = next(k for k in live if live[k].is_floating_point())
    before = shadow[key].clone()
    with torch.no_grad():
        task.model.state_dict()[key].add_(1000.0)
    moved = (shadow[key] - before).abs().max().item()
    shared = any(shadow[k].data_ptr() == live[k].data_ptr() for k in shadow)
    check(
        "1b. shadow does not alias live storage",
        moved == 0.0 and not shared,
        f"added 1000.0 to live[{key!r}]; shadow moved by {moved}; any shared data_ptr = {shared}",
    )
    with torch.no_grad():
        task.model.state_dict()[key].sub_(1000.0)  # undo

    # ---- LINK 2: evaluation actually ran on the EMA weights ----

    # Snapshot which tensors the model holds as each engine starts.
    seen = {}

    for key_name, engine in (("valid", task.engines["evaluator"]), ("raw", task.engines["raw"])):

        @engine.on(Events.STARTED)
        def _snap(_e, key_name=key_name):
            seen[key_name] = copy.deepcopy(task.model.state_dict())

    live_before_eval = copy.deepcopy(task.model.state_dict())
    ema_snapshot = {k: v.clone() for k, v in task.ema.shadow.items()}

    # run_eval_engine, not engines[...].run(): only it enters eval_context, where the swap is.
    for spec in task.eval_specs():
        task.run_eval_engine(spec)

    valid_vs_ema = max_abs_diff(seen["valid"], ema_snapshot)
    valid_vs_live = max_abs_diff(seen["valid"], live_before_eval)
    check(
        "2a. the valid engine saw the EMA weights (not the live ones)",
        valid_vs_ema == 0.0 and valid_vs_live > 1e-6,
        f"max|model_at_eval - shadow| = {valid_vs_ema}  "
        f"max|model_at_eval - live| = {valid_vs_live:.6f}",
    )

    raw_vs_live = max_abs_diff(seen["raw"], live_before_eval)
    check(
        "2b. the raw control engine saw the LIVE weights",
        raw_vs_live == 0.0,
        f"max|model_at_raw_eval - live| = {raw_vs_live}",
    )

    ema_acc = task.engines["evaluator"].state.metrics["valid/acc"]
    raw_acc = task.engines["raw"].state.metrics["raw/acc"]
    check(
        "2c. the two engines report different numbers on the same data",
        ema_acc != raw_acc,
        f"valid/acc (EMA) = {ema_acc:.4f}   raw/acc (live) = {raw_acc:.4f}   "
        f"[EMA {'beats' if ema_acc > raw_acc else 'does not beat'} raw]",
    )

    # ---- LINK 3: the live weights came back after eval ----

    after = task.model.state_dict()
    leaked = max_abs_diff(after, live_before_eval)
    check(
        "3a. live weights restored bit-exactly after eval",
        leaked == 0.0,
        f"max|model_after_eval - model_before_eval| = {leaked} "
        f"(a leak here means the next epoch trains the averaged weights)",
    )
    # The `finally`: an exception mid-eval must not leave the shadow in the model.
    boom = RuntimeError("deliberate failure mid-eval")

    @contextmanager
    def exploding_eval(spec):
        with type(task).eval_context(task, spec):
            raise boom

    before_boom = copy.deepcopy(task.model.state_dict())
    try:
        with exploding_eval(task.eval_specs()[0]):
            pass
    except RuntimeError as e:
        assert e is boom
    check(
        "3b. an exception mid-eval still restores the live weights",
        max_abs_diff(task.model.state_dict(), before_boom) == 0.0,
        f"max|model_after_exception - model_before| = "
        f"{max_abs_diff(task.model.state_dict(), before_boom)} "
        f"(without the finally, the next epoch trains the shadow)",
    )

    # ---- LINK 4: the shadow survives save -> resume ----

    from gatle_ignite.utils.checkpoints import find_latest_checkpoint

    ckpt_path = find_latest_checkpoint(cfg.save_dir)
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    check(
        "4a. the checkpoint file contains EMA state",
        "ema" in state,
        f"{ckpt_path} keys = {sorted(state.keys())}",
    )

    if "ema" not in state:
        # Fail the dependent links rather than raise: a traceback reads as a broken verifier.
        for name in (
            "4b. saved shadow is not a copy of live",
            "4c. resume restores shadow",
            "4d. resume restores the counter",
        ):
            check(name, False, "skipped: the checkpoint has no 'ema' key (see 4a)")
        summarise()
        return

    ckpt_shadow = state["ema"]["shadow"]
    ckpt_model = state["model"]
    on_disk_diff = max_abs_diff(ckpt_shadow, ckpt_model)
    check(
        "4b. the saved shadow is NOT just a copy of the saved live weights",
        on_disk_diff > 1e-6,
        f"max|ckpt.ema.shadow - ckpt.model| = {on_disk_diff:.6f}, "
        f"num_updates = {state['ema']['num_updates']}",
    )

    # A fresh trainer's shadow equals its fresh weights, so an unrestored shadow shows here.
    cfg2 = get_config()
    cfg2.max_epochs = 2
    cfg2.train_length = 30
    cfg2.save_dir = cfg.save_dir
    cfg2.resume = True
    resumed = Trainer(0, cfg2)
    resumed.setup()

    fresh_shadow = {k: v.clone() for k, v in resumed.ema.shadow.items()}
    fresh_vs_ckpt = max_abs_diff(fresh_shadow, ckpt_shadow)
    load_checkpoints(resumed)

    restored_vs_ckpt = max_abs_diff(resumed.ema.shadow, ckpt_shadow)
    restored_vs_live = max_abs_diff(resumed.ema.shadow, resumed.model.state_dict())
    check(
        "4c. resume restored the saved shadow (not a fresh copy of the live weights)",
        restored_vs_ckpt == 0.0 and fresh_vs_ckpt > 1e-6 and restored_vs_live > 1e-6,
        f"before load: max|shadow - ckpt.shadow| = {fresh_vs_ckpt:.6f} (so it was NOT already right); "
        f"after load: max|shadow - ckpt.shadow| = {restored_vs_ckpt}; "
        f"max|shadow - live| = {restored_vs_live:.6f}",
    )
    check(
        "4d. resume restored the EMA counter, so decay warmup does not restart",
        resumed.ema.num_updates == state["ema"]["num_updates"] > 0,
        f"resumed.ema.num_updates = {resumed.ema.num_updates}, "
        f"checkpoint = {state['ema']['num_updates']}",
    )

    # "best" is saved after the evaluator runs, possibly mid-swap: "model" must be the live weights.
    from gatle_ignite.utils.checkpoints import find_best_checkpoint

    best_path = find_best_checkpoint(cfg.save_dir, prefix="valid_")
    if best_path is None:
        check("4e. best checkpoint stores the LIVE weights", False, "no best checkpoint found")
    else:
        best = torch.load(best_path, map_location="cpu", weights_only=False)
        best_diff = max_abs_diff(best["model"], best["ema"]["shadow"])
        check(
            "4e. best checkpoint stores the LIVE weights, not the shadow, under 'model'",
            best_diff > 1e-6,
            f"{Path(best_path).name}: max|model - ema.shadow| = {best_diff:.6f} "
            f"(0.0 would mean the checkpoint was written mid-swap and the live weights are gone)",
        )

    summarise()


if __name__ == "__main__":
    main()
