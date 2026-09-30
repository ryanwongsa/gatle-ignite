"""Proof that two machines form ONE world. Needs PYTHONPATH=$(pwd). On every node, run:

    MASTER_ADDR=<node 0's address> python scripts/verify_multinode.py <node_rank>

Check 1 (world size) is the load-bearing one; 2 (hosts) tells one world from one box. The
metric checks catch padding and a missing reduction, never a split world. Gloo without GPUs.
"""

import os
import signal
import socket
import sys
import tempfile
from pathlib import Path

import ignite.distributed as idist
import torch

# Odd on purpose: an even split pads nothing, so a padding bug would not show.
VALID_N = 127

# Generous, since nodes start by hand. Without it a missing node hangs with no output.
RENDEZVOUS_TIMEOUT = 600

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
    print("multi-node verified: two machines, one world")


def _make_cfg(save_dir, valid_n):
    """synthetic_v0, shrunk, in a temp dir: a stale checkpoint in ckpt_dir() would be preferred."""
    from examples.synthetic.configs.synthetic_v0 import get_config

    cfg = get_config()
    cfg.save_dir = str(save_dir)
    cfg.max_epochs = 2
    cfg.logger_name = []
    cfg.train_ds_params["n"] = 256
    cfg.valid_ds_params["n"] = valid_n
    cfg.auto_model_params = {"find_unused_parameters": False, "sync_bn": False}
    return cfg


def _eval_acc(local_rank, save_dir, valid_n):
    """Evaluate the freshly seeded model, untrained, and return the metric.

    A distributed run trains a different model than one process, so only untrained runs compare.
    """
    import importlib

    cfg = _make_cfg(save_dir, valid_n)
    task = importlib.import_module(cfg.main_runner).Trainer(local_rank, cfg)
    task.setup()
    # Bypasses eval_context: safe only because synthetic defines none.
    task.engines["evaluator"].run(task.dls["valid"], max_epochs=1)
    return task.engines["evaluator"].state.metrics["valid/acc"]


def _worker(local_rank, save_dir, valid_n, results_d):
    """Runs on every rank and gathers facts; _verdict judges them in the parent.

    A raise here would surface as a traceback that reads as a broken verifier. Results cross
    nodes through the process group, since mp.Manager serves one machine; a value from another
    host is itself evidence of one group. The manager only carries it to this node's parent.
    """
    acc = _eval_acc(local_rank, save_dir, valid_n)

    # A float in, a (world_size,) tensor out.
    accs = idist.all_gather(float(acc))
    hosts = idist.all_gather(socket.gethostname())

    # Local rank 0, not global: a silent node is indistinguishable from one that never joined.
    if idist.get_local_rank() != 0:
        return
    results_d["payload"] = {
        "rank": idist.get_rank(),
        "world": idist.get_world_size(),
        "backend": idist.backend(),
        "host": socket.gethostname(),
        "hosts": list(hosts),
        "accs": [float(a) for a in accs],
    }


def _verdict(payload, baseline, expected_world):
    """Judge the gathered facts. Runs in the parent, so failures exit cleanly."""
    world = payload["world"]
    host_set = sorted(set(payload["hosts"]))
    accs = payload["accs"]

    print(
        f"\nrank {payload['rank']} of {world} on {payload['host']} | backend={payload['backend']}"
    )
    print(f"hosts gathered: {host_set}\n")

    # THE check: a node that rendezvoused with itself looks healthy in every other respect.
    check(
        "1 world spans every node",
        world == expected_world,
        f"world_size={world}, expected {expected_world}"
        + (
            ""
            if world == expected_world
            else "  <- this node formed its OWN world; it is training a separate model"
        ),
    )
    check(
        "2 the world spans >1 machine",
        len(host_set) > 1,
        f"{len(host_set)} distinct host(s): {host_set}"
        + ("" if len(host_set) > 1 else "  <- one box; node_ranks started on the same machine?"),
    )
    check(
        "3 every rank was gathered",
        len(accs) == world,
        f"gathered {len(accs)} values from a world of {world}",
    )
    uniform = len({round(a, 12) for a in accs}) == 1
    check(
        "4 the metric all-reduced (every rank agrees)",
        uniform,
        f"gathered accs: {[round(a, 6) for a in accs]}",
    )
    # Not proof of one world: a split world still shards exactly. Check 1 catches the split.
    got = accs[0]
    check(
        "5 the metric equals single-process truth",
        got == baseline,
        f"distributed={got} single-process={baseline}"
        + ("" if got == baseline else "  <- padded duplicates scored, or no reduction"),
    )


class _Timeout(Exception):
    pass


def main():
    argv = sys.argv[1:]
    if len(argv) != 1 or not argv[0].isdigit():
        raise SystemExit(
            "usage: python scripts/verify_multinode.py <node_rank>\n"
            "  Run the SAME command on every machine; node_rank is the only difference.\n"
            "  MASTER_ADDR must point at node 0 and be reachable from every node.\n"
            "  Env: MASTER_ADDR (required), MASTER_PORT (default 29500), NNODES (default 2)."
        )
    node_rank = int(argv[0])
    nnodes = int(os.environ.get("NNODES", "2"))
    addr = os.environ.get("MASTER_ADDR", "")
    port = int(os.environ.get("MASTER_PORT", "29500"))

    if nnodes < 2:
        raise SystemExit(f"NNODES={nnodes}: there is nothing multi-node to check. Use >=2.")
    if not 0 <= node_rank < nnodes:
        raise SystemExit(f"node_rank must be 0..{nnodes - 1} for NNODES={nnodes}, got {node_rank}")
    if addr in ("", "127.0.0.1", "localhost"):
        # As the CLI refuses: every node would rendezvous with itself and "pass" on its own world.
        raise SystemExit(
            f"MASTER_ADDR={addr!r} is this machine (or unset). Point it at node 0, reachable "
            f"from every node -- otherwise each node forms its own world and there is nothing "
            f"to verify. Nothing was checked."
        )

    # Rendezvous, not CUDA, is under test, so two CPU boxes will do.
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    nproc = torch.cuda.device_count() if torch.cuda.is_available() else 1
    expected_world = nnodes * nproc

    print(f"host {socket.gethostname()} | node_rank {node_rank}/{nnodes - 1}")
    print(f"backend {backend} | {nproc} proc(s)/node | expecting a world of {expected_world}")
    print(f"rendezvous tcp://{addr}:{port}\n")

    tmp = Path(tempfile.mkdtemp(prefix="verify_multinode_"))
    try:
        # Before Parallel, on each node alone: the fixed seed makes it a baseline, not gathered.
        baseline = _eval_acc(0, tmp / "baseline", VALID_N)
        print(f"single-process baseline on this node: {baseline}\n")

        os.environ["MASTER_PORT"] = str(port)
        os.environ["MASTER_ADDR"] = addr

        def _alarm(signum, frame):
            raise _Timeout()

        import torch.multiprocessing as mp

        manager = mp.Manager()
        results_d = manager.dict()

        signal.signal(signal.SIGALRM, _alarm)
        signal.alarm(RENDEZVOUS_TIMEOUT)
        try:
            # As the CLI's multi-node call: no init_method, which would double-specify it.
            with idist.Parallel(
                backend=backend,
                nproc_per_node=nproc,
                master_port=port,
                nnodes=nnodes,
                node_rank=node_rank,
                master_addr=addr,
            ) as parallel:
                parallel.run(_worker, str(tmp), VALID_N, results_d)
        except _Timeout:
            # A hang IS the result: report it as a failed check, not a traceback.
            check(
                "0 rendezvous completed",
                False,
                f"no rendezvous after {RENDEZVOUS_TIMEOUT}s at tcp://{addr}:{port}.\n"
                f"       Every node runs the same command with its own node_rank; check that\n"
                f"       the other node(s) were started, that {addr}:{port} is reachable from\n"
                f"       them (firewall/security group), and that NNODES matches everywhere.",
            )
            summarise()
        finally:
            signal.alarm(0)

        payload = results_d.get("payload")
        if payload is None:
            # The ranks came up but this node's local rank 0 never reported: not a pass.
            check(
                "0 this node reported",
                False,
                "the run finished but no rank on this node reported a result -- see any "
                "traceback above.",
            )
        else:
            _verdict(payload, baseline, expected_world)
        summarise()
    finally:
        import shutil

        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
