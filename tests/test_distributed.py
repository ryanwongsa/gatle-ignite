"""Distributed paths on CPU via gloo.

The eval-set size is deliberately not divisible by the world size: DistributedSampler pads
it, and ranks that agree on a padded split still agree on the wrong number.
"""

import ignite.distributed as idist
import pytest


def _make_cfg(save_dir, valid_n):
    from examples.synthetic.configs.synthetic_v0 import get_config

    cfg = get_config()
    cfg.save_dir = save_dir
    cfg.max_epochs = 2
    cfg.logger_name = []
    cfg.train_ds_params["n"] = 256
    cfg.valid_ds_params["n"] = valid_n
    # sync_bn requires a GPU backend.
    cfg.auto_model_params = {"find_unused_parameters": False, "sync_bn": False}
    return cfg


def _train_one(local_rank, save_dir, valid_n):
    from gatle_ignite.cli.launch import run_task

    task = run_task(local_rank, _make_cfg(save_dir, valid_n))
    return {
        "rank": idist.get_rank(),
        "world_size": idist.get_world_size(),
        "acc": task.engines["evaluator"].state.metrics["valid/acc"],
        "sampler": type(task.dls["train"].sampler).__name__,
        "eval_sampler": type(task.dls["valid"].sampler).__name__,
    }


def _worker(local_rank, save_dir, valid_n, results):
    results[idist.get_rank()] = _train_one(local_rank, save_dir, valid_n)


def _eval_only(local_rank, save_dir, valid_n, results):
    """No training: a 2-rank run trains a different model, so only eval sharding may vary."""
    import importlib

    cfg = _make_cfg(save_dir, valid_n)
    task = importlib.import_module(cfg.main_runner).Trainer(local_rank, cfg)
    task.setup()
    task.engines["evaluator"].run(task.dls["valid"], max_epochs=1)
    results[idist.get_rank()] = {
        "acc": task.engines["evaluator"].state.metrics["valid/acc"],
        "eval_sampler": type(task.dls["valid"].sampler).__name__,
    }


def _loader_batches(local_rank, _save_dir, bs, results):
    import torch
    from torch.utils.data import TensorDataset

    from gatle_ignite.data.helpers import build_dataloader

    dataset = TensorDataset(torch.arange(40))
    train = build_dataloader(dataset, {"bs": bs, "num_workers": 2})
    valid = build_dataloader(dataset, {"bs": bs, "num_workers": 2, "exact_sharding": True})
    results[idist.get_rank()] = {
        "train": [len(batch) for (batch,) in train],
        "valid": [len(batch) for (batch,) in valid],
        "valid_samples": [int(i) for (batch,) in valid for i in batch],
        "train_workers": train.num_workers,
        "valid_workers": valid.num_workers,
    }


def _run_distributed(tmp_path, valid_n, nproc=2, worker=_worker):
    import torch.multiprocessing as mp

    manager = mp.Manager()
    results = manager.dict()
    with idist.Parallel(backend="gloo", nproc_per_node=nproc) as parallel:
        parallel.run(worker, str(tmp_path), valid_n, results)
    return dict(results)


@pytest.mark.slow
def test_two_rank_gloo_shards_data_and_reduces_metrics(tmp_path):
    collected = _run_distributed(tmp_path, valid_n=128)

    assert set(collected) == {0, 1}
    assert all(r["world_size"] == 2 for r in collected.values())
    assert all(r["sampler"] == "DistributedSampler" for r in collected.values())

    # Identical across ranks means the metric was all-reduced.
    assert collected[0]["acc"] == collected[1]["acc"]


@pytest.mark.slow
def test_eval_metric_is_correct_when_the_split_does_not_divide_evenly(tmp_path):
    """127 samples over 2 ranks: a padded split would score one sample twice."""
    truth = _run_distributed(tmp_path / "single", 127, nproc=1, worker=_eval_only)[0]["acc"]
    sharded = _run_distributed(tmp_path / "double", 127, nproc=2, worker=_eval_only)

    assert sharded[0]["eval_sampler"] == "ExactDistributedSampler"
    assert sharded[0]["acc"] == sharded[1]["acc"], "metric did not all-reduce"
    assert sharded[0]["acc"] == truth, (
        f"2-rank eval reported {sharded[0]['acc']} but the true value over all 127 "
        f"samples is {truth}: the split is being padded and duplicates scored"
    )


@pytest.mark.slow
def test_bs_is_the_total_across_ranks_on_eval_splits_too(tmp_path):
    collected = _run_distributed(tmp_path, 8, worker=_loader_batches)

    for rank in (0, 1):
        assert collected[rank]["train"] == [4, 4, 4, 4, 4]
        assert collected[rank]["valid"] == collected[rank]["train"]
        assert collected[rank]["valid_workers"] == collected[rank]["train_workers"] == 1
        assert len(set(collected[rank]["valid_samples"])) == 20
    union = set(collected[0]["valid_samples"]) | set(collected[1]["valid_samples"])
    assert union == set(range(40))


@pytest.mark.slow
def test_a_bs_smaller_than_the_world_is_not_split(tmp_path):
    collected = _run_distributed(tmp_path, 1, worker=_loader_batches)

    for rank in (0, 1):
        assert collected[rank]["valid"] == [1] * 20
