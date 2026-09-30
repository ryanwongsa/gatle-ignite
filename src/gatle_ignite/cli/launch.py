"""Process launching: single-config distributed runs, and multi-config fan-out."""

import os
import random
import subprocess
import sys
import threading
from pathlib import Path

import ignite.distributed as idist
import torch

from gatle_ignite.dispatch import ConfigError, import_entrypoint


def run_task(local_rank, cfg):
    """Build the Trainer and drive it, so a user's Trainer never calls fit() itself."""
    Trainer = import_entrypoint(cfg.main_runner, "Trainer", field="main_runner")
    task = Trainer(local_rank, cfg)
    if "run" in cfg:
        # evaluate() loads its own checkpoint and enforces inference mode itself.
        task.evaluate()
        return task
    return task.fit()


def _resolve_backend(cfg):
    """A CPU-only multi-process run, such as a test, cannot use nccl."""
    if cfg.dist_backend == "nccl" and not torch.cuda.is_available():
        return "gloo"
    return cfg.dist_backend


def _parallel_kwargs(nnodes, node_rank, addr, port):
    """Never init_method: ignite >= 0.5.5 rejects a tcp:// one. Never nothing either: given no
    rendezvous argument, ignite <= 0.5.3 hardcodes port 2222 and ignores MASTER_PORT.
    """
    if nnodes > 1:
        return {
            "master_port": port,
            "nnodes": nnodes,
            "node_rank": node_rank,
            "master_addr": addr,
        }
    return {"master_addr": addr, "master_port": port}


def resolve_topology(cfg):
    """(nproc_per_node, nnodes, node_rank) for this run."""
    # Not `or 1`: that turns nnodes=0 into 1, and the check below could never fire.
    nnodes = cfg.get("nnodes", 1)
    nnodes = 1 if nnodes is None else int(nnodes)
    node_rank = cfg.get("node_rank", 0)
    node_rank = 0 if node_rank is None else int(node_rank)
    if nnodes < 1:
        raise ConfigError(f"nnodes must be >= 1, got {nnodes}")
    if not 0 <= node_rank < nnodes:
        raise ConfigError(
            f"node_rank must be in 0..{nnodes - 1} for nnodes={nnodes}, got {node_rank}"
        )

    nproc = cfg.get("nproc_per_node", None)
    if nproc is None:
        n_gpu = torch.cuda.device_count()
        nproc = n_gpu if n_gpu > 0 else 1
    nproc = int(nproc)
    if nproc < 1:
        raise ConfigError(f"nproc_per_node must be >= 1, got {nproc}")
    return nproc, nnodes, node_rank


def resolve_rendezvous(cfg, nnodes):
    """(master_addr, master_port), and refuse a multi-node run that would split apart."""
    port = cfg.get("master_port", None) or os.environ.get("MASTER_PORT")
    addr = cfg.get("master_addr", None) or os.environ.get("MASTER_ADDR") or "127.0.0.1"

    if nnodes > 1 and addr in ("127.0.0.1", "localhost"):
        raise ConfigError(
            f"nnodes={nnodes} but master_addr is {addr!r}, this machine, so each node forms "
            f"its own world and you silently train {nnodes} unrelated models instead of one.\n"
            f"  Point cfg.master_addr (or MASTER_ADDR) at node 0, reachable from every node."
        )
    if nnodes > 1 and not port:
        raise ConfigError(
            f"nnodes={nnodes} but no master_port is set, so each node picks its own random "
            f"port and the rendezvous hangs.\n"
            f"  Set cfg.master_port (or MASTER_PORT) to the same free port on every node."
        )

    port = int(port) if port else random.randint(49152, 65535)
    return addr, port


def launch(cfg):
    """Run one config, spawning one process per visible GPU."""
    nproc, nnodes, node_rank = resolve_topology(cfg)
    addr, port = resolve_rendezvous(cfg, nnodes)

    os.environ["MASTER_PORT"] = str(port)
    os.environ["MASTER_ADDR"] = addr

    if nproc * nnodes <= 1:
        # Single process: no rendezvous needed, and no backend to tear down.
        return run_task(0, cfg)

    backend = _resolve_backend(cfg)

    with idist.Parallel(
        backend=backend,
        nproc_per_node=nproc,
        **_parallel_kwargs(nnodes, node_rank, addr, port),
    ) as parallel:
        parallel.run(run_task, cfg)


def launch_many(config_paths, extra_args=None):
    """Run configs as parallel subprocesses. Each needs its own MASTER_PORT, or they collide."""
    used = set()

    def pick_port():
        while True:
            port = random.randint(49152, 65535)
            if port not in used:
                used.add(port)
                return port

    procs, threads = [], []
    for config_path in config_paths:
        prefix = Path(config_path).stem
        port = pick_port()
        print(f"Launching: {config_path} (MASTER_PORT={port})")

        env = os.environ.copy()
        # Drop inherited torchrun vars, so each child forms its own world.
        for key in (
            "RANK",
            "WORLD_SIZE",
            "LOCAL_RANK",
            "LOCAL_WORLD_SIZE",
            "GROUP_RANK",
            "ROLE_RANK",
        ):
            env.pop(key, None)
        env["MASTER_PORT"] = str(port)
        env["MASTER_ADDR"] = "127.0.0.1"

        cmd = [sys.executable, "-m", "gatle_ignite", "train", f"--config={config_path}"]
        cmd += list(extra_args or [])
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, env=env
        )
        thread = threading.Thread(target=_stream, args=(proc, prefix), daemon=True)
        thread.start()
        procs.append((prefix, proc))
        threads.append(thread)

    failed = [prefix for prefix, proc in procs if proc.wait() != 0]
    for thread in threads:
        thread.join()

    if failed:
        print(f"\nFailed configs: {', '.join(failed)}")
        return 1
    print("\nAll configs completed successfully.")
    return 0


def _stream(proc, prefix):
    for line in proc.stdout:
        sys.stdout.write(f"[{prefix}] {line}")
        sys.stdout.flush()
