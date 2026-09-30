"""Prove verify.py's checks can FAIL: each scenario breaks one link and must be caught.

Each runs in its own subprocess, since they monkeypatch module globals. Only total_leak
(no_grad and both detaches removed) actually trains the teacher, and only the accuracy
check catches `untrained`: a fresh init also differs from a never-loaded one.
"""

import subprocess
import sys

import torch.nn.functional as F

SCENARIOS = ("untrained", "submodule", "leak_grad", "total_leak")


def apply_scenario(name):
    import examples.teacher_student.losses.loss_functions.kd_loss as K
    import examples.teacher_student.trainer.student_trainer as T
    from gatle_ignite import get_value, to_device

    if name == "untrained":
        # The classic silent one: a mis-keyed state_dict load leaves random weights.
        def fake(cfg, device):
            import importlib

            m = importlib.import_module(cfg.teacher_model_name)
            t = m.Model(**dict(cfg.teacher_model_params))  # never load_state_dict'ed
            t.to(device).eval()
            for p in t.parameters():
                p.requires_grad_(False)
            return t

        T.load_teacher = fake
        return

    # The rest hang the teacher off the student: the natural mistake.
    _orig_load = T.load_teacher
    if name in ("leak_grad", "total_leak"):

        def leaky_load(cfg, device):
            t = _orig_load(cfg, device)
            for p in t.parameters():
                p.requires_grad_(True)  # the freeze forgotten
            return t

        T.load_teacher = leaky_load

    _orig_build = T.Trainer.build_model

    def build_model(self):
        model = _orig_build(self)
        model.teacher = self._teacher[0]  # nn.Module.__setattr__ registers a submodule
        return model

    T.Trainer.build_model = build_model

    if name == "total_leak":

        def leaky_prep(self, batch, split="train", **kwargs):
            x, y = batch
            out = to_device({"model_input": {"x": x}, "targets": {"labels": y}})
            out["targets"]["teacher_logits"] = self.teacher(x=out["model_input"]["x"])["logits"]
            return out

        T.Trainer.prep_batch = leaky_prep

        def leaky_kd(self, y_pred, target, iteration=None):
            s = get_value(y_pred, self.src_name)
            tl = get_value(target, self.tgt_name)
            t = self.temperature
            return F.kl_div(
                F.log_softmax(s / t, -1), F.softmax(tl / t, -1), reduction="batchmean"
            ) * (t * t)

        K.Loss.forward = leaky_kd


def main():
    if len(sys.argv) == 3 and sys.argv[1] == "--scenario":
        apply_scenario(sys.argv[2])
        import examples.teacher_student.scripts.verify as V

        rc = V.main()
        # Inverted on purpose: a scenario "succeeds" when verify FAILS.
        sys.exit(0 if rc != 0 else 1)

    print("Each scenario below must make verify.py FAIL.\n")
    results = {}
    for s in SCENARIOS:
        proc = subprocess.run(
            [sys.executable, __file__, "--scenario", s], capture_output=True, text=True
        )
        caught = proc.returncode == 0
        results[s] = caught
        failed = [ln for ln in proc.stdout.splitlines() if ln.startswith("[FAIL]")]
        print(
            f"{'[OK]  ' if caught else '[MISS]'} {s:11s} -> verify.py raised {len(failed)} failure(s)"
        )
        for ln in failed:
            print(f"           {ln}")
        print()

    missed = [s for s, ok in results.items() if not ok]
    if missed:
        print(f"verify.py FAILED TO CATCH: {missed} -- those checks are decoration.")
        return 1
    print("every scenario was caught: verify.py's checks are live.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
