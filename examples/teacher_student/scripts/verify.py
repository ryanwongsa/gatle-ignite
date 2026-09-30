"""Prove the distillation is real: each failure checked here still shows a falling loss.

1 the teacher trains too; 2 it is in the optimizer; 3 it is a fresh init, not the loaded
weights. Each check prints the number it checked, so the number can be disbelieved.
"""

import copy
import sys

import torch
from ignite.engine import Events

from examples.teacher_student.configs import base_utils as P
from examples.teacher_student.configs.teacher_student_v0 import get_config
from examples.teacher_student.trainer.student_trainer import Trainer

FAILURES = []


def check(name, ok, detail):
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}")
    if not ok:
        FAILURES.append(name)


def fingerprint(module):
    """One number that changes if ANY parameter changes."""
    with torch.no_grad():
        return float(sum((p.double() * (i + 1)).sum() for i, p in enumerate(module.parameters())))


@torch.no_grad()
def evaluate(module, loader):
    """(accuracy vs true labels, mean max-softmax) over a loader."""
    correct = total = 0
    conf = []
    for x, y in loader:
        logits = module(x=x)["logits"]
        correct += int((logits.argmax(-1) == y).sum())
        total += len(y)
        conf.append(torch.softmax(logits, -1).max(-1).values)
    return correct / total, float(torch.cat(conf).mean())


@torch.no_grad()
def agreement(a, b, loader):
    """Fraction of inputs where two models' argmaxes match."""
    same = total = 0
    for x, _ in loader:
        same += int((a(x=x)["logits"].argmax(-1) == b(x=x)["logits"].argmax(-1)).sum())
        total += len(x)
    return same / total


def main():
    cfg = get_config()
    cfg.max_epochs = 12  # a short run: these checks are about wiring, not convergence

    print("=" * 72)
    print("Building the student trainer (this loads the teacher).")
    print("=" * 72)
    task = Trainer(0, cfg)
    task.setup()

    student, teacher = task.model, task.teacher

    from examples.teacher_student.dataloaders.synthetic_dataset import get_ds

    valid_dl, _ = get_ds(dict(P.VALID))

    # ---------------------------------------------------------------- 3
    # The loaded weights are the TRAINED ones. Only accuracy proves it: a never-loaded teacher
    # also differs from `fresh` (fp 38.73 vs 24.35), so the first check passes on noise.
    print("\n--- the teacher is really trained ---")
    fresh = type(teacher)(**dict(cfg.teacher_model_params))
    same_as_fresh = fingerprint(fresh) == fingerprint(teacher)
    check(
        "teacher weights differ from a fresh random init",
        not same_as_fresh,
        f"fresh fp={fingerprint(fresh):.4f} vs loaded fp={fingerprint(teacher):.4f}",
    )

    t_acc, t_conf = evaluate(teacher, valid_dl)
    f_acc, _ = evaluate(fresh, valid_dl)
    chance = 1.0 / P.N_CLASSES
    check(
        "loaded teacher is accurate (a random init would be ~chance)",
        t_acc > 0.5 and f_acc < 0.25,
        f"loaded teacher acc={t_acc:.4f}, fresh-init acc={f_acc:.4f}, chance={chance:.4f}",
    )
    # A teacher whose soft targets are one-hot carries nothing the hard labels don't.
    check(
        "teacher's soft targets carry dark knowledge (not saturated one-hot)",
        t_conf < 0.99,
        f"mean max-softmax = {t_conf:.4f} (1.0 would mean KD == hard labels)",
    )

    # ---------------------------------------------------------------- 2
    print("\n--- the teacher is not in the optimizer ---")
    opt_ids = {id(p) for g in task.optimizer.param_groups for p in g["params"]}
    teacher_ids = {id(p) for p in teacher.parameters()}
    student_ids = {id(p) for p in student.parameters()}
    check(
        "no teacher param is in any optimizer param group",
        not (opt_ids & teacher_ids),
        f"{len(opt_ids)} params optimized, {len(teacher_ids)} teacher params, "
        f"{len(opt_ids & teacher_ids)} in common",
    )
    check(
        "the optimizer does hold the student's params",
        opt_ids and opt_ids <= student_ids,
        f"{len(opt_ids & student_ids)}/{len(student_ids)} student params optimized",
    )
    check(
        "every teacher param has requires_grad=False",
        not any(p.requires_grad for p in teacher.parameters()),
        f"{sum(p.requires_grad for p in teacher.parameters())} of {len(teacher_ids)} require grad",
    )
    check(
        "the teacher is not a submodule of the student",
        not any(m is teacher for m in student.modules()),
        f"student.state_dict() has {len(student.state_dict())} entries "
        f"(teacher alone has {len(teacher.state_dict())})",
    )

    # ---------------------------------------------------------------- 1
    print("\n--- the teacher does not change during training ---")
    before = fingerprint(teacher)
    before_sd = copy.deepcopy(teacher.state_dict())

    # Sampled inside the train loop: fit() ends in eval mode, so reading it after proves nothing.
    modes, grads = [], []

    @task.engines["trainer"].on(Events.ITERATION_COMPLETED)
    def _sample(engine):
        modes.append(teacher.training)
        grads.append(any(p.grad is not None for p in teacher.parameters()))

    task.fit()
    after = fingerprint(teacher)

    check(
        "teacher fingerprint is bit-identical after a full fit()",
        before == after,
        f"before={before!r} after={after!r}",
    )
    max_delta = max(
        float((before_sd[k].double() - v.double()).abs().max())
        for k, v in teacher.state_dict().items()
    )
    check(
        "no teacher tensor moved at all",
        max_delta == 0.0,
        f"max |delta| over every teacher tensor = {max_delta!r}",
    )
    check(
        "teacher was in eval mode on every training iteration",
        modes and not any(modes),
        f"{sum(modes)}/{len(modes)} sampled train iterations had teacher.training=True",
    )
    check(
        "no teacher param ever received a gradient",
        grads and not any(grads),
        f"{sum(grads)}/{len(grads)} sampled train iterations left a .grad on the teacher",
    )

    # The STUDENT must move, or the checks above are vacuous.
    print("\n--- ...and the student did move (or the check above proves nothing) ---")
    s_acc, _ = evaluate(student, valid_dl)
    check(
        "student trained to above chance",
        s_acc > chance * 1.5,
        f"student acc={s_acc:.4f} vs chance={chance:.4f}",
    )

    print("\n" + "=" * 72)
    if FAILURES:
        print(f"{len(FAILURES)} CHECK(S) FAILED: {FAILURES}")
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
