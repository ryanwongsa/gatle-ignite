"""The controls: a falling NT-Xent loss is not evidence, this is.

One kNN probe scores raw inputs, an untrained encoder (the positive control) and the trained
one. Both encoders go through BaseTrainer.evaluate(), the path training steers by. LEARNED
needs the trained score to beat both by the margin.
"""

import sys
import tempfile
from pathlib import Path

import torch
from ml_collections import ConfigDict

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from examples.contrastive.configs.contrastive_v0 import get_config  # noqa: E402
from examples.contrastive.dataloaders.twoview_dataset import get_ds  # noqa: E402
from examples.contrastive.metrics.knn_probe import KNNProbeAccuracy  # noqa: E402
from examples.contrastive.models.encoder import Model  # noqa: E402
from examples.contrastive.trainer.contrastive_trainer import Trainer  # noqa: E402


def _knn_score(bank_feats, bank_labels, query_feats, query_labels, k=20, temperature=0.1):
    """Score arbitrary features with the same metric object the run uses."""
    metric = KNNProbeAccuracy(lambda: (bank_feats, bank_labels), k=k, temperature=temperature)
    metric.reset()
    metric.update((query_feats, query_labels))
    return metric.compute()


def _raw_features(cfg):
    """View-1 observations straight from the datasets, with no encoder at all."""
    bank_dl, _ = get_ds(dict(cfg.bank_ds_params))
    query_dl, _ = get_ds(dict(cfg.valid_ds_params))
    bank = [torch.cat(t) for t in zip(*[(b[0], b[2]) for b in bank_dl])]
    query = [torch.cat(t) for t in zip(*[(b[0], b[2]) for b in query_dl])]
    return bank, query


def _evaluate_ckpt(cfg, ckpt_dir):
    """Score the checkpoint in ckpt_dir through the framework's own eval path.

    evaluate() needs inference mode and then loads the checkpoint on every call, so weights
    can only be scored by writing them to a directory and pointing save_dir at it.
    """
    cfg = ConfigDict(cfg.to_dict())
    cfg.run = True
    cfg.load_from_ckpt = "best"
    cfg.save_ckpt = False
    cfg.logger_name = []
    cfg.save_dir = str(ckpt_dir)

    trainer = Trainer(0, cfg)
    trainer.setup()
    return trainer.evaluate()["valid/knn"]


def _write_untrained_ckpt(cfg, trained_ckpt, out_dir):
    """A fresh-init encoder, written in the same format, so it loads down the same path."""
    state = torch.load(trained_ckpt, map_location="cpu", weights_only=False)
    fresh = Model(**cfg.model_params).state_dict()

    ref = "encoder.0.weight"
    assert not torch.allclose(state["model"][ref], fresh[ref]), (
        "the trained checkpoint equals a fresh init: the control would not be a control"
    )

    state["model"] = fresh
    out = out_dir / "valid_best_result_checkpoint_0_0.0000.pt"
    torch.save(state, out)
    return out


def main():
    cfg = get_config()
    knn = dict(cfg.knn_params)
    knn.pop("src_name", None)

    best = sorted(Path(cfg.save_dir).glob("valid_best_result_checkpoint_*.pt"))
    if not best:
        print(
            "no checkpoint: run `gatle-ignite train --config=examples/contrastive/configs/contrastive_v0.py`"
        )
        return
    best = max(best, key=lambda p: float(p.stem.rsplit("_", 1)[-1]))

    torch.manual_seed(cfg.seed)

    # --- raw-input baseline: no encoder at all ------------------------------
    (bank_x, bank_y), (query_x, query_y) = _raw_features(cfg)
    raw = _knn_score(bank_x, bank_y, query_x, query_y, **knn)

    # --- untrained encoder: the positive control ----------------------------
    with tempfile.TemporaryDirectory() as tmp:
        _write_untrained_ckpt(cfg, best, Path(tmp))
        untrained = _evaluate_ckpt(cfg, Path(tmp))

    # --- trained encoder ----------------------------------------------------
    trained = _evaluate_ckpt(cfg, Path(cfg.save_dir))

    chance = 1.0 / cfg.valid_ds_params["n_classes"]
    print("\n=== kNN probe on frozen features (same bank, same queries, same metric) ===")
    print(f"chance                      : {chance:.4f}")
    print(f"raw inputs (no encoder)     : {raw:.4f}")
    print(f"untrained encoder (control) : {untrained:.4f}")
    print(f"trained encoder             : {trained:.4f}   [{best.name}]")

    margin = 0.05
    ok = trained > max(raw, untrained) + margin
    print(f"\nverdict: {'LEARNED' if ok else 'NO EVIDENCE'}")
    print(
        f"  trained beats raw inputs by {trained - raw:+.4f} and the untrained "
        f"encoder by {trained - untrained:+.4f} (need > {margin} over both)"
    )
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
