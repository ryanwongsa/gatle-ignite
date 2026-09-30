"""Proof that a multi-GPU run is correct. Needs >=2 GPUs, and PYTHONPATH=$(pwd).

1 EXACT SHARDING: an uneven eval split is not padded; its metric equals 1-process truth.
2 SYNC_BN: the shipped default, sync_bn=True, really converts a model's BatchNorm.
3 AUTOCAST: bf16 and fp16 each get the right GradScaler, learn, and compute in that dtype.
Checks 2 and 3 cannot run on CPU, so were never forced to fail: if one does, suspect this script.
"""

import shutil
import socket
import tempfile
from pathlib import Path

import ignite
import ignite.distributed as idist
import torch

PASS, FAIL = "PASS", "FAIL"
results = []

# Odd on purpose: an even split pads nothing, so check 1 could not fail.
VALID_N = 127
BN_MODEL = "examples.ema.models.mlp"  # same signature as synthetic's, but with BatchNorm1d

# synthetic_v0's defaults, which train loss 2.31 -> 0.59. The ceiling sits clear of 0.59 for
# bf16 numerics, and well under the ~2.31 an untrained model scores.
TRAIN_EPOCHS = 15
TRAIN_N = 2048
LOSS_REFERENCE = 0.59
LOSS_CEILING = 1.0


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
    print("multi-GPU verified: nccl, exact sharding, sync_bn, autocast")


def _make_cfg(
    save_dir, valid_n, model_name=None, amp_dtype="fp32", sync_bn=False, max_epochs=2, train_n=256
):
    """synthetic_v0, resized. The 2-epoch/256-sample default is too small to learn anything.

    save_dir is a temp dir, never ckpt_dir(): a leftover real checkpoint there would be preferred.
    """
    from examples.synthetic.configs.synthetic_v0 import get_config

    cfg = get_config()
    cfg.save_dir = str(save_dir)
    cfg.max_epochs = max_epochs
    cfg.logger_name = []
    cfg.train_ds_params["n"] = train_n
    cfg.valid_ds_params["n"] = valid_n
    cfg.amp_dtype = amp_dtype
    if model_name:
        cfg.model_name = model_name
    cfg.auto_model_params = {"find_unused_parameters": False, "sync_bn": sync_bn}
    return cfg


def _run(save_dir, valid_n, backend, nproc, worker, **kw):
    """Single node only: mp.Manager() serves from this machine, which other hosts can't reach."""
    import torch.multiprocessing as mp

    manager = mp.Manager()
    results_d = manager.dict()
    with idist.Parallel(backend=backend, nproc_per_node=nproc) as parallel:
        parallel.run(worker, str(save_dir), valid_n, results_d, **kw)
    return dict(results_d)


def _eval_only(local_rank, save_dir, valid_n, results_d, model_name=None):
    """Evaluate the freshly seeded model, untrained.

    A 2-rank run trains a different model than 1 process (sharded data, divided batch size),
    so trained runs would differ on correct code. Untrained, only the sharding differs.
    """
    import importlib

    cfg = _make_cfg(save_dir, valid_n, model_name=model_name)
    task = importlib.import_module(cfg.main_runner).Trainer(local_rank, cfg)
    task.setup()
    # Bypasses eval_context: safe only because synthetic defines none. For a config that
    # does (examples/ema), switch to run_eval_engine.
    task.engines["evaluator"].run(task.dls["valid"], max_epochs=1)
    results_d[idist.get_rank()] = {
        "acc": task.engines["evaluator"].state.metrics["valid/acc"],
        "eval_sampler": type(task.dls["valid"].sampler).__name__,
        "world_size": idist.get_world_size(),
        "device": str(idist.device()),
    }


def _train_probe(
    local_rank,
    save_dir,
    valid_n,
    results_d,
    model_name=None,
    amp_dtype="fp32",
    sync_bn=False,
    max_epochs=2,
    train_n=256,
):
    """Train, then report what the trainer actually built, reached, and computed in."""
    import importlib

    cfg = _make_cfg(
        save_dir,
        valid_n,
        model_name=model_name,
        amp_dtype=amp_dtype,
        sync_bn=sync_bn,
        max_epochs=max_epochs,
        train_n=train_n,
    )
    # By hand, not run_task, so forward can be wrapped. setup() is idempotent, so fit() matches.
    task = importlib.import_module(cfg.main_runner).Trainer(local_rank, cfg)
    task.setup()

    # The evidence autocast is real: forward runs inside torch.autocast, so its output must
    # come back bf16/fp16. A disabled autocast still descends and builds the right scaler.
    seen = {}
    _forward = task.forward

    def forward_spy(model_input):
        out = _forward(model_input)
        tensor = out["logits"] if isinstance(out, dict) else out
        seen.setdefault("forward_dtype", str(tensor.dtype))
        return out

    task.forward = forward_spy
    task.fit()

    results_d[idist.get_rank()] = {
        "has_sync_bn": any(isinstance(m, torch.nn.SyncBatchNorm) for m in task.model.modules()),
        "has_scaler": task.scaler is not None,
        "loss": task.engines["trainer"].state.metrics["train/loss_avg"],
        "acc": task.engines["evaluator"].state.metrics["valid/acc"],
        "world_size": idist.get_world_size(),
        "forward_dtype": seen.get("forward_dtype", "<forward never ran>"),
        "configured_dtype": str(task.dtype),
        "autocast_enabled": task.autocast_enabled,
        "autocast_device": task.device_type,
    }


def main():
    if not torch.cuda.is_available():
        raise SystemExit("no CUDA: this script needs >=2 GPUs. Nothing was checked.")
    n_gpu = torch.cuda.device_count()
    if n_gpu < 2:
        raise SystemExit(f"found {n_gpu} GPU(s); this script needs >=2. Nothing was checked.")

    print(f"host {socket.gethostname()} | {n_gpu} GPUs | {torch.cuda.get_device_name(0)}")
    print(f"torch {torch.__version__} | ignite {ignite.__version__} | cuda {torch.version.cuda}\n")

    tmp = Path(tempfile.mkdtemp(prefix="verify_multigpu_"))
    try:
        # ---- 1: the eval split is sharded exactly, and the metric is TRUE -------------
        # Both legs on nccl/GPU: fp32 kernels differ across devices, which is not under test.
        truth = _run(tmp / "single", VALID_N, "nccl", 1, _eval_only)[0]["acc"]
        sharded = _run(tmp / "double", VALID_N, "nccl", 2, _eval_only)

        check(
            "1a nccl formed a 2-rank world",
            sharded[0]["world_size"] == 2 and sharded[0]["device"].startswith("cuda"),
            f"world_size={sharded[0]['world_size']} device={sharded[0]['device']}",
        )
        check(
            "1b eval split sharded exactly",
            sharded[0]["eval_sampler"] == "ExactDistributedSampler",
            f"{VALID_N} samples over 2 ranks -> {sharded[0]['eval_sampler']}",
        )
        # Necessary, not sufficient: ranks also agree on a padded, wrong answer. 1d has teeth.
        check(
            "1c metric all-reduced (ranks agree)",
            sharded[0]["acc"] == sharded[1]["acc"],
            f"rank0={sharded[0]['acc']} rank1={sharded[1]['acc']}",
        )
        # Forced to fail on CPU/gloo with exact_sharding=False: 1c agreed, 1d gave 10/128.
        check(
            "1d metric equals single-process truth",
            sharded[0]["acc"] == truth,
            f"2-rank={sharded[0]['acc']} 1-proc={truth}"
            + ("" if sharded[0]["acc"] == truth else "  <- padded duplicates scored"),
        )

        # ---- 2: sync_bn, the shipped default, actually converts ------------------------
        # Short run: this inspects the model the trainer built, which need not learn.
        bn = _run(
            tmp / "syncbn", VALID_N, "nccl", 2, _train_probe, model_name=BN_MODEL, sync_bn=True
        )
        check(
            "2a sync_bn=True converts BatchNorm to SyncBatchNorm",
            bn[0]["has_sync_bn"],
            f"model contains SyncBatchNorm: {bn[0]['has_sync_bn']} (model={BN_MODEL})",
        )
        # The control that keeps 2a honest: synthetic's MLP has no BatchNorm to convert.
        plain = _run(tmp / "plainbn", VALID_N, "nccl", 2, _train_probe, sync_bn=True)
        check(
            "2b control: BN-free model gets no SyncBatchNorm",
            not plain[0]["has_sync_bn"],
            f"synthetic MLP contains SyncBatchNorm: {plain[0]['has_sync_bn']} (must be False)",
        )

        # ---- 3: autocast ---------------------------------------------------------------
        # Loss descent, not metric equality: bf16 can legitimately flip a prediction, so do
        # not "tighten" it. Full-size training, since the question is whether it still learns.
        for dtype, want_scaler in (("bf16", False), ("fp16", True)):
            got = _run(
                tmp / dtype,
                VALID_N,
                "nccl",
                2,
                _train_probe,
                amp_dtype=dtype,
                max_epochs=TRAIN_EPOCHS,
                train_n=TRAIN_N,
            )
            check(
                f"3 {dtype}: GradScaler {'built' if want_scaler else 'not built'}",
                got[0]["has_scaler"] is want_scaler,
                f"scaler={got[0]['has_scaler']}, expected {want_scaler}"
                " (a scaler is only meaningful for fp16)",
            )
            check(
                f"3 {dtype}: loss descends",
                got[0]["loss"] < LOSS_CEILING,
                f"train/loss_avg={got[0]['loss']:.4f} after {TRAIN_EPOCHS} epochs"
                f" (start ~2.31, fp32 reference {LOSS_REFERENCE}, ceiling {LOSS_CEILING})",
            )
            # The one with teeth: only a live autocast makes the model's output bf16/fp16.
            want = f"torch.{'bfloat16' if dtype == 'bf16' else 'float16'}"
            check(
                f"3 {dtype}: the forward really computed in {dtype}",
                got[0]["forward_dtype"] == want,
                f"model output dtype={got[0]['forward_dtype']}, expected {want}"
                f"  [configured={got[0]['configured_dtype']}, "
                f"enabled={got[0]['autocast_enabled']}, device={got[0]['autocast_device']}]"
                + (
                    ""
                    if got[0]["forward_dtype"] == want
                    else "  <- autocast did not engage; the loss and scaler checks above "
                    "cannot see this"
                ),
            )
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    summarise()


if __name__ == "__main__":
    main()
