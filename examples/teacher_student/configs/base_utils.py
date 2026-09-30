"""Shared shapes and paths, so the teacher and student configs cannot drift."""

from pathlib import Path

# configs/base_utils.py -> configs/ -> the project root.
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Stage 1's subprocess starts here, where `examples.` paths resolve. A standalone project
# would use PROJECT_ROOT.
REPO_ROOT = PROJECT_ROOT.parents[1]

# Relative to REPO_ROOT, where the subprocess runs.
TEACHER_CONFIG = "examples/teacher_student/configs/teacher_v0.py"

IN_DIM = 32
N_CLASSES = 8

TEACHER_DIR = PROJECT_ROOT / "checkpoints" / "teacher"
STUDENT_DIR = PROJECT_ROOT / "checkpoints" / "student"

# The teacher is wide; the student is narrow. That gap is the point of distilling.
TEACHER_PARAMS = {"in_dim": IN_DIM, "hidden": 256, "depth": 2, "n_classes": N_CLASSES}
STUDENT_PARAMS = {"in_dim": IN_DIM, "hidden": 16, "depth": 1, "n_classes": N_CLASSES}

DS_COMMON = {"in_dim": IN_DIM, "n_classes": N_CLASSES, "num_workers": 0}

# The student gets a small, DISJOINT slice (another seed): on the teacher's own rows its
# logits are near one-hot, and the soft term would add nothing.
TEACHER_TRAIN = {**DS_COMMON, "n": 16384, "seed": 0, "bs": 256, "shuffle": True, "drop_last": True}
STUDENT_TRAIN = {**DS_COMMON, "n": 256, "seed": 7, "bs": 64, "shuffle": True, "drop_last": True}
VALID = {**DS_COMMON, "n": 2048, "seed": 99, "bs": 512, "shuffle": False}


def teacher_ckpt():
    """The best-scoring stage-1 checkpoint, by the score in its name: the newest is not the best."""
    cands = list(TEACHER_DIR.glob("valid_best_result_checkpoint_*.pt"))
    if not cands:
        return str(TEACHER_DIR / "MISSING_train_the_teacher_first.pt")
    return str(max(cands, key=lambda p: float(p.stem.split("_")[-1])))
