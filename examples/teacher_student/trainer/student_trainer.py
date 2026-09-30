"""The student's trainer. It owns the teacher, a second model the framework must never train.

build_model loads the teacher beside the student; prep_batch puts its logits in `targets`.
"""

import importlib
import os
import shutil
import subprocess
from pathlib import Path

import torch

from examples.teacher_student.configs.base_utils import REPO_ROOT, TEACHER_CONFIG
from gatle_ignite import BaseTrainer, load_weights, to_device


def _autotrain_teacher(repo_root):
    """Run stage 1 in a SUBPROCESS if its checkpoint is missing.

    In-process, its fit() would reseed the RNGs, so the student's run would depend on
    whether the teacher already existed.
    """
    print(f"[teacher_student] no teacher checkpoint yet: training stage 1 ({TEACHER_CONFIG})")
    # The console script the docs lead with, so a failure shows where a reader would look.
    exe = shutil.which("gatle-ignite")
    if exe is None:
        raise RuntimeError(
            "the 'gatle-ignite' CLI is not on PATH, so the teacher cannot be trained "
            f"automatically. Train it yourself:\n  gatle-ignite train --config={TEACHER_CONFIG}"
        )
    env = {
        **os.environ,
        "PYTHONPATH": os.pathsep.join(filter(None, [str(repo_root), os.environ.get("PYTHONPATH")])),
    }
    proc = subprocess.run([exe, "train", f"--config={TEACHER_CONFIG}"], cwd=repo_root, env=env)
    if proc.returncode != 0:
        raise RuntimeError(
            "could not train the teacher automatically. Run it yourself:\n"
            f"  gatle-ignite train --config={TEACHER_CONFIG}"
        )


def load_teacher(cfg, device):
    """Build the teacher and load stage-1 weights into it. Frozen, eval, off the tape."""
    module = importlib.import_module(cfg.teacher_model_name)
    teacher = module.Model(**dict(cfg.teacher_model_params))

    path = Path(cfg.teacher_weights)
    if not path.exists() and cfg.get("teacher_autotrain", False):
        _autotrain_teacher(REPO_ROOT)
        # The filename carries the score, so it is not known until stage 1 has run.
        from examples.teacher_student.configs import base_utils as P

        path = Path(P.teacher_ckpt())
    if not path.exists():
        raise FileNotFoundError(
            f"teacher checkpoint not found: {path}\n"
            f"Train the teacher first:  gatle-ignite train --config={TEACHER_CONFIG}"
        )

    # Not cfg.model_checkpoint_dir, which only loads the student. strict=True: a mis-keyed
    # load would leave a random teacher and a run that still looks plausible.
    load_weights(path, teacher, strict=True)

    teacher.to(device).eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    print(
        f"[teacher_student] teacher loaded from {path.name} "
        f"({sum(p.numel() for p in teacher.parameters())} params, frozen, eval)"
    )
    return teacher


class Trainer(BaseTrainer):
    def build_model(self):
        model = super().build_model()  # the student: dotted dispatch + idist.auto_model

        import ignite.distributed as idist

        # In a LIST: were this ever on an nn.Module, a plain attribute would register the
        # teacher into the optimizer, every checkpoint, and model.train().
        self._teacher = [load_teacher(self.cfg, idist.device())]
        return model

    @property
    def teacher(self):
        return self._teacher[0]

    def prep_batch(self, batch, split="train", **kwargs):
        x, y = batch
        out = to_device({"model_input": {"x": x}, "targets": {"labels": y}})

        # Every split: training needs the logits for the KD term, eval for teacher_agreement.
        with torch.no_grad():
            teacher_logits = self.teacher(x=out["model_input"]["x"])["logits"]
        # .detach() on top of no_grad: the KD term can never reach back into the teacher.
        out["targets"]["teacher_logits"] = teacher_logits.detach()
        return out
