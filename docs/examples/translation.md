# Speech translation (seq2seq, variable length)

Variable-length "audio" features in one toy language, text tokens out in another. An encoder-decoder
transformer, 173k params. CPU, no downloads, ~36 s.

```bash
gatle-ignite train --config=examples/translation/configs/translation_v0.py
python -m examples.translation.scripts.checks                       # the proof, run after training
gatle-ignite eval --config=examples/translation/configs/translation_v0.py --ckpt best
```

<!-- spine: translation/translation_v0 -->

## The task is not a copy

Each of 12 source words has a prototype vector, uttered as 2-4 noisy frames. An utterance is 2-4
words, so sources are 4-16 frames and targets 4-6 tokens: **variable on both sides**. The reference
translation is deterministic but reversed and permuted:

```python
target = [BOS] + [PERM[w] for w in reversed(src_words)] + [EOS]
```

The reversal forces real cross-attention. A monotonic frame-*i*-to-token-*i* aligner cannot score on
this, which is what makes a good result meaningful.

## What this example demonstrates

- **A custom `collate_fn`:** pass it through `ds_params` and `build_dataloader` forwards it. It
  survives both DDP sharding paths, so a custom collate and exact eval sharding are orthogonal.
- **Masked loss:** `ignore_index=PAD`, so padding never trains toward `<pad>`.
- **Train/eval asymmetry:** teacher forcing while training, greedy autoregressive decode at eval.
  This needed **zero** step overrides: `prep_batch` reads `split`, and the model branches on whether
  it was given a decoder input.
- **Lower-is-better scoring:** WER selects the best checkpoint via `score_factor = -1`.

```python title="examples/translation/trainer/translation_trainer.py"
--8<-- "examples/translation/trainer/translation_trainer.py"
```

## The evidence

A falling teacher-forced loss proves nothing: a model can teacher-force well and decode into garbage.
So the metrics score the **decoded** output.

```text
epoch 1   valid/exact 0.191   valid/wer 0.4170
epoch 7   valid/exact 0.926   valid/wer 0.0270      <- best
```

- **Positive control**: the untrained model scores `wer 1.8381` (above 1, because it emits garbage
  insertions) and `exact 0.0000`.
- **Padding really is masked**: perturbing the logits by `25*N(0,1)` **at pad positions only** moves
  the loss by exactly `0.00e+00`. The control that makes this non-vacuous: the same perturbation on an
  unmasked loss moves it `2.938 -> 11.543`. Gradients at PAD are `0.000e+00`; at real positions
  `7.904e-03`.
- **The decode is not secretly teacher-forced**: rolling the source within the batch collapses exact
  match `0.8984 -> 0.0078`.

!!! note "Two ways a check like these can lie"
    To test masking, perturb the pad *logits*, not the pad *labels*: changing a label stops that
    position being ignored and renormalises `reduction="mean"`. And to find the best checkpoint by
    filename, take the **largest** score, not the smallest: the number in the filename is
    `score_factor * metric`, so with `score_factor = -1` the best is the least negative value.
