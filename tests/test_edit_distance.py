"""The translation example's WER/CER metric."""

import pytest
import torch
from examples.translation.metrics.edit_distance import EditDistance, edit_distance, get_metric

from gatle_ignite import ConfigError


@pytest.mark.parametrize(
    "hyp,ref,expected",
    [
        ([1, 2, 3], [1, 2, 3], 0),
        ([1, 2, 3], [1, 2, 4], 1),  # substitution
        ([1, 2], [1, 2, 3], 1),  # deletion
        ([1, 2, 3, 4], [1, 2, 3], 1),  # insertion
        ([], [1, 2, 3], 3),  # empty hypothesis costs the whole reference
        ([1, 2, 3], [], 3),  # and vice versa
        ([], [], 0),
        ([4, 3, 2, 1], [1, 2, 3, 4], 4),  # reversal
    ],
)
def test_known_edit_distances(hyp, ref, expected):
    assert edit_distance(hyp, ref) == expected


def _fed(rows, mode="rate", **kwargs):
    metric = EditDistance(mode=mode, **kwargs)
    metric.reset()
    for pred, tgt in rows:
        metric.update((torch.tensor([pred]), torch.tensor([tgt])))
    return metric.compute()


def test_a_perfect_decode_scores_zero():
    assert _fed([([1, 2, 3], [1, 2, 3])]) == 0.0


def test_the_rate_is_distance_over_reference_length():
    assert _fed([([1, 2, 9], [1, 2, 3])]) == pytest.approx(1 / 3)


def test_the_rate_is_a_corpus_rate_not_a_mean_of_rates():
    """A mean of per-utterance rates would be (1/1 + 0/9)/2 = 0.5; the corpus rate is 0.1."""
    rows = [([9], [1]), (list(range(2, 11)), list(range(2, 11)))]
    assert _fed(rows) == pytest.approx(0.1)


def test_wer_and_cer_are_aliases_of_rate():
    rows = [([1, 2, 9], [1, 2, 3])]
    assert _fed(rows, mode="wer") == _fed(rows, mode="cer") == _fed(rows, mode="rate")


def test_exact_match_is_not_the_rate():
    rows = [([1, 2, 3], [1, 2, 3]), ([1, 2, 9], [1, 2, 3])]
    assert _fed(rows, mode="exact") == pytest.approx(0.5)
    assert _fed(rows, mode="rate") == pytest.approx(1 / 6)


def test_padding_and_eos_are_stripped():
    assert _fed([([1, 2, 3, 0, 0], [1, 2, 3, 0, 0])], pad_id=0) == 0.0
    assert _fed([([1, 2, 2, 9, 9], [1, 2, 2, 9, 9])], eos_id=9) == 0.0
    # eos truncates: everything after it is decoder noise, not a prediction.
    assert _fed([([1, 2, 9, 7, 7], [1, 2, 9])], eos_id=9) == 0.0


def test_an_all_empty_reference_raises_rather_than_scoring_perfect():
    """0/0 reported as 0.0 would read as a flawless model; it is a broken pad_id."""
    with pytest.raises(ValueError, match="every reference is empty"):
        _fed([([1, 2], [0, 0])], pad_id=0)


def test_no_examples_raises():
    metric = EditDistance()
    metric.reset()
    with pytest.raises(ValueError, match="at least one example"):
        metric.compute()


def test_a_bad_mode_names_the_alternatives():
    with pytest.raises(ConfigError, match="mode must be one of"):
        EditDistance(mode="werr")


def test_get_metric_reads_pad_and_eos_from_info():
    metric = get_metric("valid", {"pad_id": 5, "eos_id": 6})
    assert (metric.pad_id, metric.eos_id) == (5, 6)


def test_an_explicit_param_beats_info():
    metric = get_metric("valid", {"pad_id": 5, "eos_id": 6}, pad_id=0, eos_id=2)
    assert (metric.pad_id, metric.eos_id) == (0, 2)


def test_get_metric_reads_the_named_keys():
    """Through output_transform, as ignite applies it: update() alone sees the raw dict."""
    metric = get_metric("valid", {}, src_name="hyp", tgt_name=("targets", "ref"))
    step_output = {
        "y_pred": {"hyp": torch.tensor([[1, 2, 9]])},
        "target": {"targets": {"ref": torch.tensor([[1, 2, 3]])}},
    }
    metric.reset()
    metric.update(metric._output_transform(step_output))
    assert metric.compute() == pytest.approx(1 / 3)


def test_it_accepts_plain_lists_as_well_as_tensors():
    metric = EditDistance(pad_id=None)
    metric.reset()
    metric.update(([[1, 2, 9]], [[1, 2, 3]]))
    assert metric.compute() == pytest.approx(1 / 3)


# ---- DDP ----------------------------------------------------------------------


def _ddp_worker(local_rank, results):
    import ignite.distributed as idist_
    from examples.translation.metrics.edit_distance import EditDistance

    # Rank 0 sees a perfect utterance, rank 1 a wrong one.
    shard = [([1, 2, 3], [1, 2, 3])] if idist_.get_rank() == 0 else [([1, 2, 9], [1, 2, 3])]

    metric = EditDistance(mode="wer", device=idist_.device())
    metric.reset()
    for pred, tgt in shard:
        metric.update((torch.tensor([pred]), torch.tensor([tgt])))
    results[idist_.get_rank()] = metric.compute()


@pytest.mark.slow
def test_the_metric_reduces_across_ranks(tmp_path):
    import ignite.distributed as idist_
    import torch.multiprocessing as mp

    manager = mp.Manager()
    results = manager.dict()
    with idist_.Parallel(backend="gloo", nproc_per_node=2) as parallel:
        parallel.run(_ddp_worker, results)

    collected = dict(results)
    assert set(collected) == {0, 1}
    # Pin the value: ranks can agree on a wrong number.
    assert collected[0] == pytest.approx(1 / 6), (
        f"expected the corpus rate over BOTH shards (1/6); got {collected[0]}: a "
        f"rank-local metric would report 0.0 on rank 0 and 1/3 on rank 1"
    )
    assert collected[0] == collected[1]
