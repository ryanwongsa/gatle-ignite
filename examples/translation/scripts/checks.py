"""Proof the example works: padding is masked from the loss, and the real decode is right."""

import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from examples.translation.configs.translation_v0 import get_config
from examples.translation.dataloaders.data_utils.vocab import EOS_ID, PAD_ID
from examples.translation.dataloaders.toy_speech_dataset import ToySpeechTranslation, collate_fn
from examples.translation.losses.loss_functions.masked_ce import Loss
from examples.translation.metrics.edit_distance import _strip
from examples.translation.metrics.edit_distance import edit_distance as _edit_distance
from examples.translation.models.seq2seq import Model

CFG = get_config()


def _valid_loader(bs=128):
    p = dict(CFG.valid_ds_params)
    ds = ToySpeechTranslation(
        n=p["n"], in_dim=p["in_dim"], seed=p["seed"], task_seed=p["task_seed"]
    )
    return DataLoader(ds, batch_size=bs, shuffle=False, collate_fn=collate_fn)


def _build_model():
    return Model(**dict(CFG.model_params))


def _best_ckpt():
    ckpts = sorted(Path(CFG.save_dir).glob("valid_best_result_checkpoint_*.pt"))
    if not ckpts:
        return None

    # Filenames hold score_factor * wer = -wer, so the best is the MAXIMUM, not the newest.
    def score(p):
        return float(p.stem.split("_")[-1])

    return max(ckpts, key=score)


def _load_trained():
    path = _best_ckpt()
    if path is None:
        print(f"NO CHECKPOINT in {CFG.save_dir}, train first:")
        print(
            "  PYTHONPATH=$(pwd) gatle-ignite train --config=examples/translation/configs/translation_v0.py"
        )
        sys.exit(1)
    model = _build_model()
    state = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(state["model"])
    model.eval()
    return model, path


def check_padding_is_masked():
    """The loss must not depend on ANYTHING at padded target positions.

    Perturbs only the pad positions' LOGITS: the masked loss must not move, an unmasked
    control must. Perturbing labels would change the ignored count and move the loss anyway.
    """
    print("\n=== check_padding_is_masked ===")
    torch.manual_seed(0)
    model = _build_model().eval()
    batch = next(iter(_valid_loader(bs=64)))

    tgt_out = batch["tgt_out"]
    pad_positions = tgt_out == PAD_ID
    frac = pad_positions.float().mean().item()
    print(
        f"padded target positions in batch: {pad_positions.sum().item()}/{tgt_out.numel()} ({frac:.1%})"
    )
    assert frac > 0.05, "batch has almost no padding: this check would prove nothing"

    with torch.no_grad():
        logits = model(batch["src_feats"], batch["src_pad_mask"], batch["tgt_in"])["logits"]

    noise = torch.zeros_like(logits)
    noise[pad_positions] = 25.0 * torch.randn_like(noise[pad_positions])
    corrupted = logits + noise
    assert not torch.equal(corrupted, logits), "corruption was a no-op"
    assert torch.equal(corrupted[~pad_positions], logits[~pad_positions]), (
        "corrupted real positions too"
    )

    tgt = {"targets": {"tgt_out": tgt_out}}
    masked = Loss(ignore_index=PAD_ID)
    a = masked({"logits": logits}, tgt).item()
    b = masked({"logits": corrupted}, tgt).item()
    print(f"masked loss   clean={a:.6f}  pad-logits-corrupted={b:.6f}  delta={abs(a - b):.2e}")
    assert a == b, f"MASKING BROKEN: loss moved by {abs(a - b)} when only <pad> logits changed"

    # The control: without masking, the same corruption must move the loss.
    unmasked = Loss(ignore_index=-100)
    c = unmasked({"logits": logits}, tgt).item()
    d = unmasked({"logits": corrupted}, tgt).item()
    print(f"unmasked loss clean={c:.6f}  pad-logits-corrupted={d:.6f}  delta={abs(c - d):.2e}")
    assert abs(c - d) > 1e-4, "the control did not move: this test cannot detect broken masking"

    # ignore_index must actually exclude positions.
    print(f"masked={a:.6f} vs unmasked={c:.6f} on identical data (must differ)")
    assert abs(a - c) > 1e-4, "masked and unmasked agree: ignore_index is doing nothing"
    print("PASS: padding is excluded from the loss, and the check can fail.")


def check_grad_does_not_flow_to_pad():
    """A sharper version: no gradient may reach the logits at padded positions."""
    print("\n=== check_grad_does_not_flow_to_pad ===")
    torch.manual_seed(0)
    model = _build_model()
    batch = next(iter(_valid_loader(bs=64)))
    logits = model(batch["src_feats"], batch["src_pad_mask"], batch["tgt_in"])["logits"]
    logits.retain_grad()
    Loss(ignore_index=PAD_ID)(
        {"logits": logits}, {"targets": {"tgt_out": batch["tgt_out"]}}
    ).backward()

    pad_positions = batch["tgt_out"] == PAD_ID
    g = logits.grad.abs().sum(-1)
    print(f"max |grad| at PAD target positions: {g[pad_positions].max().item():.3e}")
    print(f"max |grad| at real target positions: {g[~pad_positions].max().item():.3e}")
    assert g[pad_positions].max().item() == 0.0, "gradient reaches padded positions"
    assert g[~pad_positions].max().item() > 0.0, "no gradient anywhere: check is vacuous"
    print("PASS: gradient is exactly zero at padded positions and non-zero elsewhere.")


@torch.no_grad()
def _score_decode(model, loader):
    """Greedy AR decode over the whole loader -> (wer, exact). No teacher forcing."""
    dist = ref_len = exact = n = 0
    for batch in loader:
        # tgt_in deliberately NOT passed: this is the eval path the trainer uses.
        pred = model(batch["src_feats"], batch["src_pad_mask"])["pred_tokens"]
        for p_row, t_row in zip(pred, batch["tgt_out"]):
            hyp = _strip(p_row.tolist(), PAD_ID, EOS_ID)
            ref = _strip(t_row.tolist(), PAD_ID, EOS_ID)
            dist += _edit_distance(hyp, ref)
            ref_len += len(ref)
            exact += hyp == ref
            n += 1
    return dist / ref_len, exact / n


def check_autoregressive_decode():
    """The real claim: the trained model DECODES the right targets, beside an untrained control."""
    print("\n=== check_autoregressive_decode ===")
    loader = _valid_loader()

    torch.manual_seed(0)
    untrained = _build_model().eval()
    u_wer, u_exact = _score_decode(untrained, loader)
    print(f"UNTRAINED (positive control): wer={u_wer:.4f}  exact={u_exact:.4f}")

    trained, path = _load_trained()
    t_wer, t_exact = _score_decode(trained, loader)
    print(f"TRAINED  ({path.name}): wer={t_wer:.4f}  exact={t_exact:.4f}")

    assert u_wer > 0.7, (
        f"untrained control is suspiciously good (wer={u_wer:.4f}): task may be trivial"
    )
    assert t_wer < 0.15, f"trained model does not decode (wer={t_wer:.4f})"
    assert t_exact > 0.80, (
        f"trained model rarely decodes an utterance exactly right ({t_exact:.4f})"
    )
    print(f"PASS: decode wer {u_wer:.4f} -> {t_wer:.4f}, exact {u_exact:.4f} -> {t_exact:.4f}")


def check_decode_is_not_teacher_forcing():
    """Shuffling the source must wreck the decode, or it is reading the target from somewhere."""
    print("\n=== check_decode_is_not_teacher_forcing ===")
    model, _ = _load_trained()
    batch = next(iter(_valid_loader(bs=128)))

    with torch.no_grad():
        ok = model(batch["src_feats"], batch["src_pad_mask"])["pred_tokens"]
        perm = torch.roll(torch.arange(batch["src_feats"].shape[0]), 1)
        shuffled = model(batch["src_feats"][perm], batch["src_pad_mask"][perm])["pred_tokens"]

    def acc(pred):
        hits = sum(
            _strip(p.tolist(), PAD_ID, EOS_ID) == _strip(t.tolist(), PAD_ID, EOS_ID)
            for p, t in zip(pred, batch["tgt_out"])
        )
        return hits / len(batch["tgt_out"])

    a, b = acc(ok), acc(shuffled)
    print(f"exact-match with matching source={a:.4f}  with mismatched source={b:.4f}")
    assert a > 0.8 and b < 0.1, "decode does not depend on the source: it is not translating"
    print("PASS: the decode depends on the source audio, as a translation must.")


def show_examples(k=5):
    """Eyeball the actual decodes, including the word-order reversal."""
    print("\n=== sample decodes (trained, greedy AR) ===")
    model, _ = _load_trained()
    batch = next(iter(_valid_loader(bs=k)))
    with torch.no_grad():
        pred = model(batch["src_feats"], batch["src_pad_mask"])["pred_tokens"]
    for i in range(k):
        hyp = _strip(pred[i].tolist(), PAD_ID, EOS_ID)
        ref = _strip(batch["tgt_out"][i].tolist(), PAD_ID, EOS_ID)
        frames = int((~batch["src_pad_mask"][i]).sum())
        print(f"  {frames:2d} frames -> hyp={hyp}  ref={ref}  {'ok' if hyp == ref else 'WRONG'}")


if __name__ == "__main__":
    check_padding_is_masked()
    check_grad_does_not_flow_to_pad()
    check_autoregressive_decode()
    check_decode_is_not_teacher_forcing()
    show_examples()
    print("\nALL CHECKS PASSED")
