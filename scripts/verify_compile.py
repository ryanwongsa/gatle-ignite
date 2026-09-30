"""Proof that `cfg.compile` is correct on a GPU, which no CI leg has. Needs PYTHONPATH=$(pwd).

The CNN is built with dropout 0, so eager and compiled draw the same masks. Without CUDA,
6 (fp16 + GradScaler) skips, and 7 (DDP) skips below 2 GPUs. A skip is never a pass.
"""

import shutil
import tempfile
from pathlib import Path

import torch

PASS, FAIL, SKIP = "PASS", "FAIL", "SKIP"
results = []


def _provenance():
    """(commit, subject) of this checkout, so a pasted report names the code it ran."""
    import subprocess

    def git(*args):
        try:
            return subprocess.run(
                ["git", *args], capture_output=True, text=True, timeout=10, check=False
            ).stdout.strip()
        except Exception:
            return ""

    sha = git("rev-parse", "--short", "HEAD") or "unknown"
    subject = git("log", "-1", "--format=%s")
    dirty = " +uncommitted changes" if git("status", "--porcelain") else ""
    return f"{sha}{dirty}", subject


def check(name, ok, detail):
    results.append((name, ok, detail))
    print(f"[{PASS if ok else FAIL}] {name}\n       {detail}")


def skip(name, why):
    results.append((name, None, why))
    print(f"[{SKIP}] {name}\n       {why}")


def summarise():
    print("\n" + "=" * 62)
    ran = [(n, ok) for n, ok, _ in results if ok is not None]
    failed = [n for n, ok in ran if not ok]
    skipped = [n for n, ok, _ in results if ok is None]
    print(
        f"{len(ran) - len(failed)}/{len(ran)} checks passed"
        + (f", {len(skipped)} skipped" if skipped else "")
    )
    if skipped:
        print("SKIPPED: " + ", ".join(skipped))
    if failed:
        print("FAILED: " + ", ".join(failed))
        raise SystemExit(1)
    commit, _ = _provenance()
    print(f"commit {commit}")
    print("Paste this whole output where the run was asked for, with the GPU model and driver.")


def _cnn(device):
    from examples.classification.models.shapes_cnn import Model

    return Model(n_classes=5, width=16, dropout=0.0).to(device)


def _batch(device, batch=8):
    image = torch.randn(batch, 1, 32, 32, device=device)
    target = torch.randint(0, 5, (batch,), device=device)
    return image, target


def check_guard(device):
    import torch.nn as nn

    from gatle_ignite.dispatch import ConfigError
    from gatle_ignite.enhancements.compile import compile_model

    try:
        compile_model(nn.Sequential(nn.Linear(4, 4)).to(device))
        check("1 guard refuses a model compile cannot reach", False, "no ConfigError raised")
    except ConfigError as e:
        check(
            "1 guard refuses a model compile cannot reach",
            "Sequential" in str(e),
            f"ConfigError names the class: {str(e).splitlines()[0][:80]}",
        )


def check_checkpoints(device, tmp):
    from ignite.handlers import Checkpoint

    from gatle_ignite.enhancements.compile import compile_model

    torch.manual_seed(0)
    compiled = compile_model(_cnn(device))
    torch.manual_seed(1)
    eager = _cnn(device)

    keys = list(compiled.state_dict())
    check(
        "2 compiled state_dict carries no _orig_mod. prefix",
        not any(k.startswith("_orig_mod.") for k in keys),
        f"{len(keys)} keys, first: {keys[0]}",
    )

    state = {"model": compiled.state_dict()}
    Checkpoint.load_objects(to_load={"model": eager}, checkpoint=state)
    same = sum(
        torch.allclose(v, eager.state_dict()[k].to(v.dtype))
        for k, v in compiled.state_dict().items()
    )
    check(
        "3 a compiled checkpoint loads into an eager model",
        same == len(keys),
        f"{same}/{len(keys)} tensors match after load",
    )

    path = Path(tmp) / "compiled.pt"
    torch.save(state, path)
    torch.manual_seed(2)
    target = compile_model(_cnn(device))
    Checkpoint.load_objects(
        to_load={"model": target}, checkpoint=torch.load(path, map_location=device)
    )
    same = sum(
        torch.allclose(v, target.state_dict()[k].to(v.dtype))
        for k, v in compiled.state_dict().items()
    )
    check(
        "4 the same checkpoint reloads into a compiled model",
        same == len(keys),
        f"{same}/{len(keys)} tensors match after a save/load round trip on {device.type}",
    )


def check_amp(device):
    import torch._dynamo as dynamo
    import torch.nn as nn

    image, target = _batch(device)
    criterion = nn.CrossEntropyLoss()

    def loss_of(compiled):
        torch.manual_seed(0)
        dynamo.reset()
        model = _cnn(device)
        if compiled:
            model.compile()
        model.train()
        with torch.autocast(device.type, dtype=torch.bfloat16):
            logits = model(image=image)["logits"]
        return criterion(logits.float(), target).item()

    eager, compiled = loss_of(False), loss_of(True)
    delta = abs(eager - compiled)
    check(
        "5 bf16 loss parity, compiled vs eager",
        delta < 5e-3,
        f"eager {eager:.6f} vs compiled {compiled:.6f}, delta {delta:.2e} "
        f"(CPU max 2.3e-4 over 12 input seeds; bound is 5e-3)",
    )


def check_fp16(device):
    """fp16's GradScaler is built only on CUDA, so no CI leg reaches it.

    A full step, not a forward: the scaler's whole job is in the backward and the step.
    """
    if device.type != "cuda":
        skip("6 fp16 + GradScaler under compile", "a GradScaler is only built on CUDA")
        return

    import torch._dynamo as dynamo
    import torch.nn as nn

    image, target = _batch(device)
    criterion = nn.CrossEntropyLoss()

    def one_step(compiled):
        torch.manual_seed(0)
        dynamo.reset()
        model = _cnn(device)
        if compiled:
            model.compile()
        model.train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        scaler = torch.amp.GradScaler("cuda")
        with torch.autocast("cuda", dtype=torch.float16):
            logits = model(image=image)["logits"]
            loss = criterion(logits.float(), target)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        grad = next(p.grad for p in model.parameters() if p.grad is not None)
        return loss.item(), scaler.get_scale(), torch.isfinite(grad).all().item()

    try:
        eager_loss, eager_scale, eager_finite = one_step(False)
        comp_loss, comp_scale, comp_finite = one_step(True)
    except Exception as e:
        check("6 fp16 + GradScaler under compile", False, f"{type(e).__name__}: {str(e)[:120]}")
        return

    delta = abs(eager_loss - comp_loss)
    check(
        "6 fp16 + GradScaler under compile",
        delta < 5e-3 and eager_finite and comp_finite,
        f"eager {eager_loss:.6f} vs compiled {comp_loss:.6f}, delta {delta:.2e} "
        f"(bf16 on CPU: max 2.3e-4 over 12 input seeds; bound is 5e-3); "
        f"grads finite eager={eager_finite} compiled={comp_finite}; "
        f"scale {eager_scale:g} -> {comp_scale:g}",
    )


def check_ddp():
    if torch.cuda.device_count() < 2:
        skip(
            "7 DDP + compile trains on 2 GPUs",
            f"needs >=2 CUDA devices, found {torch.cuda.device_count()}. The gloo equivalent "
            f"runs in CI (tests/test_compile.py) and covers the code path but not nccl.",
        )
        return

    import ignite.distributed as idist
    import torch.multiprocessing as mp

    tmp = tempfile.mkdtemp(prefix="verify_compile_ddp_")
    try:
        manager = mp.Manager()
        collected = manager.dict()
        with idist.Parallel(backend="nccl", nproc_per_node=2) as parallel:
            parallel.run(_ddp_worker, tmp, collected)
        got = dict(collected)
        ok = set(got) == {0, 1} and all(
            r["compiled"] and r["world_size"] == 2 for r in got.values()
        )
        check(
            "7 DDP + compile trains on 2 GPUs",
            ok,
            f"ranks {sorted(got)}, compiled={[r['compiled'] for r in got.values()]}, "
            f"loss={[round(r['loss'], 4) for r in got.values()]}",
        )
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def _ddp_worker(local_rank, save_dir, collected):
    import ignite.distributed as idist
    from examples.synthetic.configs.synthetic_v0 import get_config

    from gatle_ignite.cli.launch import run_task
    from gatle_ignite.enhancements.compile import is_compiled

    cfg = get_config()
    cfg.save_dir = save_dir
    cfg.max_epochs = 1
    cfg.logger_name = []
    cfg.train_ds_params["n"] = 512
    cfg.valid_ds_params["n"] = 256
    cfg.compile = True
    task = run_task(local_rank, cfg)
    collected[idist.get_rank()] = {
        "world_size": idist.get_world_size(),
        "compiled": is_compiled(task.model),
        "loss": task.engines["trainer"].state.metrics.get("train/loss_avg", float("nan")),
    }


def check_auto(device):
    """CI has no accelerator, so it only ever sees "auto" resolve OFF."""
    from ml_collections import config_dict

    from gatle_ignite.enhancements.compile import resolve_compile

    cfg = config_dict.ConfigDict()
    cfg.compile = "auto"
    compiling = resolve_compile(cfg)
    check(
        "8 cfg.compile = auto resolves ON",
        compiling is (device.type == "cuda"),
        f"auto -> {compiling} on {device.type} (expected {device.type == 'cuda'})",
    )


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    commit, subject = _provenance()
    print(f"commit      {commit}  {subject[:60]}")
    print(f"device={device} torch={torch.__version__} cuda_devices={torch.cuda.device_count()}")
    if device.type != "cuda":
        print(
            "\nNo CUDA device. This script will still run, but it then proves nothing CI does "
            "not already prove -- the point is the hardware. Checks 6 and 7 will skip.\n"
        )

    tmp = tempfile.mkdtemp(prefix="verify_compile_")
    try:
        check_guard(device)
        check_checkpoints(device, tmp)
        check_amp(device)
        check_fp16(device)
        check_ddp()
        check_auto(device)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    summarise()


if __name__ == "__main__":
    main()
