"""WER, CER and exact match. Score DECODED output: a model can teacher-force well and decode badly."""

import torch
from ignite.metrics import Metric
from ignite.metrics.metric import reinit__is_reduced, sync_all_reduce

from gatle_ignite import ConfigError, get_value

_RATE_ALIASES = ("rate", "wer", "cer")


def edit_distance(hyp, ref):
    """Levenshtein distance between two sequences."""
    if not hyp:
        return len(ref)
    if not ref:
        return len(hyp)
    prev = list(range(len(ref) + 1))
    for i, h in enumerate(hyp, start=1):
        cur = [i] + [0] * len(ref)
        for j, r in enumerate(ref, start=1):
            cur[j] = min(
                prev[j] + 1,  # deletion
                cur[j - 1] + 1,  # insertion
                prev[j - 1] + (h != r),  # substitution
            )
        prev = cur
    return prev[-1]


def _strip(seq, pad_id, eos_id):
    """Trim a padded, EOS-terminated row of ids down to the real tokens."""
    out = []
    for token in seq:
        token = int(token)
        if eos_id is not None and token == eos_id:
            break
        if pad_id is not None and token == pad_id:
            continue
        out.append(token)
    return out


class EditDistance(Metric):
    """Edit distance over reference length ("rate"), or the exact-match fraction ("exact").

    A corpus rate, not a mean of per-utterance rates, so a one-word reference weighs less.
    """

    def __init__(
        self,
        mode="rate",
        pad_id=0,
        eos_id=None,
        output_transform=lambda x: x,
        device="cpu",
    ):
        if mode not in (*_RATE_ALIASES, "exact"):
            raise ConfigError(
                f"mode must be one of {(*_RATE_ALIASES, 'exact')}, got {mode!r}.\n"
                f"  Use 'wer' for word tokens, 'cer' for characters, 'exact' for whole matches."
            )
        self.mode = "rate" if mode in _RATE_ALIASES else mode
        self.pad_id = pad_id
        self.eos_id = eos_id
        super().__init__(output_transform=output_transform, device=device)

    @reinit__is_reduced
    def reset(self):
        self._dist = torch.tensor(0.0, device=self._device)
        self._ref_len = torch.tensor(0.0, device=self._device)
        self._exact = torch.tensor(0.0, device=self._device)
        self._n = torch.tensor(0.0, device=self._device)
        super().reset()

    @reinit__is_reduced
    def update(self, output):
        pred, tgt = output
        for pred_row, tgt_row in zip(pred, tgt):
            hyp = _strip(_as_list(pred_row), self.pad_id, self.eos_id)
            ref = _strip(_as_list(tgt_row), self.pad_id, self.eos_id)
            self._dist += edit_distance(hyp, ref)
            self._ref_len += len(ref)
            self._exact += float(hyp == ref)
            self._n += 1

    @sync_all_reduce("_dist", "_ref_len", "_exact", "_n")
    def compute(self):
        if int(self._n) == 0:
            raise ValueError("EditDistance needs at least one example before it can compute")
        if self.mode == "exact":
            return (self._exact / self._n).item()
        if int(self._ref_len) == 0:
            # 0/0. Reporting 0.0 would read as a perfect score for a broken target.
            raise ValueError(
                "every reference is empty, so an edit-distance rate is undefined (0/0). "
                "Check pad_id/eos_id: stripping the wrong id empties every reference."
            )
        return (self._dist / self._ref_len).item()


def _as_list(row):
    return row.tolist() if isinstance(row, torch.Tensor) else list(row)


def get_metric(
    engine_type,
    info,
    src_name="pred_tokens",
    tgt_name=("targets", "tokens"),
    mode="rate",
    pad_id=None,
    eos_id=None,
    **kwargs,
):
    """`pad_id` and `eos_id` default to the dataset's `info`; a value in the config wins."""
    info = info or {}

    def output_transform(output):
        return get_value(output["y_pred"], src_name), get_value(output["target"], tgt_name)

    return EditDistance(
        mode=mode,
        pad_id=info.get("pad_id", 0) if pad_id is None else pad_id,
        eos_id=info.get("eos_id", None) if eos_id is None else eos_id,
        output_transform=output_transform,
        **kwargs,
    )
