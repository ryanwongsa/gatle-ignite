"""Proof that each wired-up piece fires: every check asserts on a number it observed."""

import sys

import torch

from examples.classification.configs.classification_v0 import get_config
from examples.classification.dataloaders.data_utils.balanced_sampler import get_sampler
from examples.classification.dataloaders.data_utils.shapes import CLASS_NAMES, N_CLASSES
from examples.classification.dataloaders.shapes_dataset import get_ds
from gatle_ignite import get_aug

PASS, FAIL = "PASS", "FAIL"
_results = []


def check(name, ok, detail):
    _results.append((name, bool(ok)))
    print(f"[{PASS if ok else FAIL}] {name}\n       {detail}")


def _labels_of(dl):
    return torch.cat([y for _, y in dl])


# ---------------------------------------------------------------- 1. imbalance is real
cfg = get_config()
train_dl, _ = get_ds({**cfg.train_ds_params, "sampler_params": None}, transform=None)
raw = torch.bincount(_labels_of(train_dl), minlength=N_CLASSES)
check(
    "train split is genuinely imbalanced",
    raw.max().item() >= 10 * raw.min().item(),
    f"raw class counts {dict(zip(CLASS_NAMES, raw.tolist()))}, ratio "
    f"{raw.max().item() / raw.min().item():.0f}:1",
)

valid_raw = torch.bincount(_labels_of(get_ds(cfg.valid_ds_params, None)[0]), minlength=N_CLASSES)
test_raw = torch.bincount(_labels_of(get_ds(cfg.test_ds_params, None)[0]), minlength=N_CLASSES)
check(
    "valid is balanced (it picks the checkpoint) but test keeps the deployment skew",
    bool((valid_raw == valid_raw[0]).all()) and test_raw.max() >= 10 * test_raw.min(),
    f"valid {valid_raw.tolist()} | test {test_raw.tolist()}",
)

# ------------------------------------------------- 2. the sampler actually rebalances
ds, _ = get_ds({**cfg.train_ds_params, "sampler_params": None}, transform=None)
dataset = ds.dataset
sampler = get_sampler(dataset, num_samples=20000, num_classes=N_CLASSES)
drawn = torch.bincount(dataset.labels[torch.tensor(list(sampler))], minlength=N_CLASSES)
frac = drawn.float() / drawn.sum()
check(
    "sampler rebalances the classes it draws",
    bool(((frac - 1.0 / N_CLASSES).abs() < 0.02).all()),
    f"20000 draws -> {dict(zip(CLASS_NAMES, [round(f, 3) for f in frac.tolist()]))} "
    f"(uniform = {1.0 / N_CLASSES:.3f}); raw fractions were "
    f"{[round(f, 3) for f in (raw.float() / raw.sum()).tolist()]}",
)

# ------------------------- 3. the sampler is the one the CONFIG actually wires in
sampled_dl, _ = get_ds(cfg.train_ds_params, transform=None)
check(
    "the config's sampler_params reach the real dataloader",
    sampled_dl.sampler is not None and type(sampled_dl.sampler).__name__ == "WeightedRandomSampler",
    f"train dataloader sampler = {type(sampled_dl.sampler).__name__}, "
    f"num_samples={len(sampled_dl.sampler)}",
)

epoch = torch.bincount(_labels_of(sampled_dl), minlength=N_CLASSES)
check(
    "one real epoch through that loader is near-uniform",
    bool(((epoch.float() / epoch.sum() - 1.0 / N_CLASSES).abs() < 0.05).all()),
    f"epoch counts {dict(zip(CLASS_NAMES, epoch.tolist()))} vs raw {raw.tolist()}",
)

# ------------------------------- 4. the augmentation reaches TRAIN and NOT eval
transform = get_aug(cfg.aug_name, cfg.aug_params)
train_dl2, _ = get_ds(cfg.train_ds_params, transform=transform)
valid_dl, _ = get_ds(cfg.valid_ds_params, transform=transform)
test_dl, _ = get_ds(cfg.test_ds_params, transform=transform)

before = transform.calls
_ = _labels_of(train_dl2)
after_train = transform.calls
_ = _labels_of(valid_dl)
_ = _labels_of(test_dl)
after_eval = transform.calls

check(
    "augmentation is applied on the train split",
    after_train > before,
    f"transform.calls {before} -> {after_train} over one train epoch "
    f"({after_train - before} samples augmented)",
)
check(
    "augmentation is NOT applied on valid/test",
    after_eval == after_train,
    f"transform.calls stayed at {after_eval} across a full valid + test pass",
)

# The transform must actually change pixels, not just get called.
torch.manual_seed(0)
x0 = dataset.x[0]
deltas = [(transform(x0) - x0).abs().max().item() for _ in range(20)]
check(
    "augmentation actually perturbs the image",
    min(deltas) > 1e-6,
    f"max|aug(x)-x| over 20 draws: min={min(deltas):.3f}, max={max(deltas):.3f}",
)

# Two reads of the SAME eval index must be identical, or eval is nondeterministic.
valid_ds = valid_dl.dataset
a, b = valid_ds[3][0], valid_ds[3][0]
check(
    "eval samples are deterministic across reads",
    torch.equal(a, b),
    f"valid[3] read twice: max abs diff = {(a - b).abs().max().item():.1e}",
)

# ------------------------------------------------ 5. the metrics discriminate at all
# A constant metric passes 'above chance' by luck, so feed it a known-degenerate prediction.
from examples.classification.metrics.per_class import get_metric as per_class_metric  # noqa: E402
from examples.classification.metrics.topk import get_metric as topk_metric  # noqa: E402

info = {"length": 500, "num_classes": N_CLASSES, "class_names": list(CLASS_NAMES)}
y = torch.tensor([0, 0, 0, 0, 1])  # 4x class 0, 1x class 1
logits = torch.zeros(5, N_CLASSES)
logits[:, 0] = 1.0  # always predict class 0
out = {"y_pred": {"logits": logits}, "target": {"targets": {"labels": y}}}

m = per_class_metric("valid", info)
m.reset()
m.update(m._output_transform(out))
bal = m.compute()
check(
    "balanced accuracy punishes a majority-only model (plain acc would not)",
    abs(bal - 0.5) < 1e-6,
    f"always-predict-class-0 on 4:1 data -> bal_acc={bal:.3f} (plain acc would be 0.800)",
)

logits2 = torch.tensor([[0.0, 1.0, 0.5, 0.0, 0.0]])  # true=0 is 3rd best
mt = topk_metric("valid", info, k=2)
mt.reset()
mt.update(
    mt._output_transform(
        {"y_pred": {"logits": logits2}, "target": {"targets": {"labels": torch.tensor([0])}}}
    )
)
top2_miss = mt.compute()
mt.reset()
mt.update(
    mt._output_transform(
        {"y_pred": {"logits": logits2}, "target": {"targets": {"labels": torch.tensor([2])}}}
    )
)
top2_hit = mt.compute()
check(
    "top2 is a real top-2 (hits the 2nd-ranked class, misses the 3rd)",
    top2_miss == 0.0 and top2_hit == 1.0,
    f"true=rank3 -> {top2_miss:.1f}; true=rank2 -> {top2_hit:.1f}",
)

# ------------------------------------------- 6. the LR schedule warms up, then anneals
# A scheduler that never fires leaves the LR flat, silently, so drive the real one.
from ignite.engine import Engine  # noqa: E402

from gatle_ignite.schedulers import prep_scheduler  # noqa: E402

sched_model = torch.nn.Linear(4, 2)
opt = torch.optim.AdamW(sched_model.parameters(), lr=cfg.optimizer_params["lr"])
scheduler, engine_name, event = prep_scheduler(
    cfg.lr_scheduler, cfg, sampled_dl, opt, {"trainer": Engine(lambda e, b: None)}
)
steps_per_epoch = len(sampled_dl)
lrs = []
for _ in range(cfg.max_epochs * steps_per_epoch):
    scheduler(None)
    lrs.append(opt.param_groups[0]["lr"])

warm_end = steps_per_epoch * cfg.lr_scheduler_params["warmup_epochs"]
peak = max(lrs)
check(
    "LR warms up linearly, peaks at the configured lr, then cosine-anneals down",
    lrs[0] < 0.5 * peak
    and abs(peak - cfg.optimizer_params["lr"]) < 1e-9
    and abs(lrs[warm_end - 1] - peak) < 1e-9
    and lrs[-1] < 0.05 * peak,
    f"lr: start={lrs[0]:.2e} -> peak={peak:.2e} at step {lrs.index(peak)} "
    f"(warmup ends at {warm_end}) -> end={lrs[-1]:.2e}; attached to '{engine_name}' on {event}",
)

# ------------------------------------------------------- 7. three engines are declared
from examples.classification.trainer.classification_trainer import Trainer  # noqa: E402

cfg_probe = get_config()
cfg_probe.max_epochs = 1
specs = Trainer(0, cfg_probe).eval_specs()
splits = {s.split for s in specs}
check(
    "valid AND test engines are declared (three engines with the trainer)",
    {"valid", "test"} <= splits and cfg_probe.every_test > 0,
    f"eval_specs -> {sorted(splits)}; every_val={cfg_probe.every_val}, "
    f"every_test={cfg_probe.every_test}",
)

print()
failed = [n for n, ok in _results if not ok]
print(f"{len(_results) - len(failed)}/{len(_results)} checks passed")
if failed:
    print("FAILED: " + ", ".join(failed))
sys.exit(1 if failed else 0)
