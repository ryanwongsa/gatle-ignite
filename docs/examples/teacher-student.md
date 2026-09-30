# Teacher-student (a second, frozen model)

Knowledge distillation: a trained teacher supervises a much smaller student through a KL term on
temperature-scaled logits. CPU, no downloads, ~30 s for both stages.

```bash
gatle-ignite train --config=examples/teacher_student/configs/teacher_student_v0.py
python examples/teacher_student/scripts/verify.py            # the proof
python examples/teacher_student/scripts/negative_controls.py # proves the proof can fail
```

Stage 1 trains the teacher (76,296 params) and auto-runs in a subprocess if its checkpoint is
missing. Stage 2 trains the student (664 params) on a small **disjoint** train set.

**Stage 1, the teacher:** `teacher_v0.py`

<!-- spine: teacher_student/teacher_v0 -->

**Stage 2, the student:** `teacher_student_v0.py`

<!-- spine: teacher_student/teacher_student_v0 -->

## The pattern: the teacher rides in `targets`

Every framework contract passes `(y_pred, target)`. So `prep_batch` runs the teacher under `no_grad`
and drops its logits into `targets`, after which the KD term is an ordinary config-wired sub-loss
reading `tgt_name = ("targets", "teacher_logits")`. Nothing bespoke.

The teacher is built in `build_model()` and held in a **list** (`self._teacher = [m]`) so that
`nn.Module.__setattr__` cannot auto-register it as a submodule, which would put it in the
checkpoint, in the optimizer, and on the gradient tape.

```python title="examples/teacher_student/trainer/student_trainer.py"
--8<-- "examples/teacher_student/trainer/student_trainer.py"
```

## The evidence

Distillation must beat the hard-label baseline, and the teacher must genuinely be frozen. Same seed,
so both runs share an init and a data order; the control zeroes the KD weight through `--override`:

| | valid/acc | valid/teacher_agreement |
|---|---|---|
| teacher | 0.7915 | n/a |
| **student + KD** | **0.5850** | **0.6050** |
| student, hard labels only | 0.4976 | 0.5063 |
| chance | 0.1250 | n/a |

Agreement is the load-bearing number: it rises on *held-out* data only if the student is being pulled
toward the teacher's function. The control logs `train/loss_kd_avg` of exactly `0.0`, so it really is
hard-label-only.

The teacher is proven frozen from *inside* the train loop: bit-identical fingerprint across `fit()`,
0/48 iterations in train mode, 0/48 iterations with a gradient, and 0 of its params in any optimizer
group.

## Two lessons about checks

- **"The teacher's weights differ from a fresh init" is worthless.** It passes against a teacher that
  was never loaded, because an untrained teacher is just a *different* random draw. Only accuracy
  catches it (`loaded teacher acc=0.7915` vs `fresh-init acc=0.1001`, chance 0.1250).
- **Check eval mode from inside the loop, not after `fit()`.** By the time `fit()` returns, its
  closing eval pass has called `.eval()` and put everything back, so reading `teacher.training` then
  passes even for a teacher that was in train mode for **48/48** training iterations.

!!! warning "`model_checkpoint_dir` will not load your teacher"
    It targets the **primary** model, the one named by `model_name`. Pointing it at a teacher
    checkpoint tries to load the teacher into the student. Here the shapes differ so it raises; for
    self-distillation or a mean teacher, **where the architectures match, it would load silently and
    do the wrong thing.** Load a second model's weights yourself in `build_model()`.
