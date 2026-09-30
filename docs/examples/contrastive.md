# Contrastive / self-supervised (SimCLR)

Two augmented views per sample, NT-Xent loss, kNN probe on frozen features. CPU, no downloads, ~15 s.

```bash
gatle-ignite train --config=examples/contrastive/configs/contrastive_v0.py
python examples/contrastive/scripts/probe_check.py     # the proof
```

<!-- spine: contrastive/contrastive_v0 -->

Two things that sound hard here are not:

- **A dataset returning two views.** `prep_batch` maps them to two model inputs. No special support.
- **A loss with no labels.** NT-Xent's targets are *positions in the batch*, so the sub-loss simply
  reads two `src_*` names and no `tgt_name` at all. Nothing requires a loss to have a target.

```python title="examples/contrastive/losses/loss_functions/ntxent.py"
--8<-- "examples/contrastive/losses/loss_functions/ntxent.py"
```

## Why the loss is not the evidence

NT-Xent falls for degenerate representations too. The evidence is a **kNN probe on frozen features,
against labels the loss never sees**, built imperatively via `dict_metric_from_list`, because it has
to close over the live model.

```text
chance                      : 0.1000
raw inputs (no encoder)     : 0.4004
untrained encoder (control) : 0.3643
trained encoder             : 0.9756
verdict: LEARNED
```

The untrained encoder scoring **below** raw inputs (0.364 vs 0.400) is what a working control should
show: a random projection destroys information. `probe_check.py` exits non-zero on NO EVIDENCE.

## Three ways an SSL result can fool you

Worth reading if you are evaluating SSL, because each one looks like success:

1. **A task that is too easy.** At `style_scale = 1.5`, `valid/knn` reaches 1.0 at epoch 1, but raw
   inputs and the untrained encoder score 1.000 too: a falling loss and 100% accuracy, proving
   nothing. This example uses `style_scale = 4.0`.
2. **A control that loads the trained weights.** Inference mode evaluates what is on disk, not what
   is in memory (see [`evaluate()`](../reference/api.md)), so an "untrained" control run through it
   loads the trained checkpoint and scores the same as the trained model. That reads as a plausible
   "SSL didn't work" result, when the numbers being identical is the clue.
3. **Random draws in the collate.** They make the eval set non-deterministic, so one checkpoint
   scores differently across runs. Every random draw here happens in the dataset.
