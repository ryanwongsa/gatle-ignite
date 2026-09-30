"""Every contract has a template, and every template that can share one config trains in it.

Two cannot: a backend is a sink, trained in test_logging.py, and a collate would need its own
dataset and prep_batch, so examples/templates/collate.py is only import-checked.
"""

import pytest
import torch

from gatle_ignite import base_config
from gatle_ignite.dispatch import CONTRACTS, import_entrypoint

# module -> the entrypoint its contract requires it to expose.
TEMPLATES = {
    "examples.templates.trainer": "Trainer",
    "examples.templates.model": "Model",
    "examples.templates.dataset": "get_ds",
    "examples.templates.augmentation": "Transformation",
    "examples.templates.loss": "Loss",
    "examples.templates.optimizer": "get_optimizer",
    "examples.templates.scheduler": "get_scheduler",
    "examples.templates.metric": "get_metric",
    "examples.templates.sampler": "get_sampler",
    "examples.templates.collate": "get_collate_fn",
    "examples.templates.backend": "Backend",
}


@pytest.mark.parametrize("module,attr", sorted(TEMPLATES.items()))
def test_template_exposes_its_entrypoint(module, attr):
    import_entrypoint(module, attr, field="template")


def test_every_contract_has_a_template():
    """The contracts table says what a module exposes; only a template shows what to write."""
    required = {attr for _, attr, _, _ in CONTRACTS}
    covered = set(TEMPLATES.values())
    assert not required - covered, (
        f"contracts with no template: {sorted(required - covered)}. "
        "Add examples/templates/<thing>.py and list it in TEMPLATES."
    )
    assert not covered - required, (
        f"templates for things that are not contracts: {sorted(covered - required)}"
    )


@pytest.fixture
def cfg(tmp_path):
    """A config wired from every template that can share one (see the module docstring)."""
    cfg = base_config()
    cfg.name = "templates"
    cfg.project_name = "gatle-templates"
    cfg.save_dir = str(tmp_path)
    cfg.logger_name = []
    cfg.amp_dtype = "fp32"
    cfg.max_epochs = 2

    cfg.main_runner = "examples.templates.trainer"
    cfg.model_name = "examples.templates.model"
    cfg.model_params = {"in_dim": 64, "hidden": 32, "n_classes": 10}

    cfg.aug_name = "examples.templates.augmentation"
    cfg.aug_params = {"p": 0.5, "scale": 0.01}

    common = {"n": 128, "bs": 32}
    cfg.train_ds_name = "examples.templates.dataset"
    cfg.train_ds_params = {
        **common,
        "split": "train",
        # No `shuffle`: the sampler owns the order.
        "sampler_params": {
            "cls_name": "examples.templates.sampler",
            "params": {"num_samples": 128},
        },
    }
    cfg.valid_ds_name = "examples.templates.dataset"
    cfg.valid_ds_params = {**common, "split": "valid"}

    # The trainer template declares a "probe" engine, so its fields are required.
    cfg.probe_ds_name = "examples.templates.dataset"
    cfg.probe_ds_params = {**common, "split": "valid"}
    cfg.every_probe = 1

    cfg.criterion_name = "gatle_ignite.losses.composite"
    cfg.criterion_params = {
        "dict_of_loss_params": {
            "ce": {
                "cls_name": "examples.synthetic.losses.loss_functions.cross_entropy",
                "loss_params": {"src_name": "logits", "tgt_name": ("targets", "labels")},
                "weight": 1.0,
            },
            "focal": {
                "cls_name": "examples.templates.loss",
                "loss_params": {"src_name": "logits", "tgt_name": ("targets", "labels")},
                "weight": 0.1,
            },
        }
    }

    cfg.optimizer_name = "examples.templates.optimizer"
    cfg.optimizer_params = {"lr": 1e-3, "weight_decay": 0.01}
    cfg.lr_scheduler = "examples.templates.scheduler"
    cfg.lr_scheduler_params = {"step_every_epochs": 1, "gamma": 0.5}

    metric = {
        "cls_name": "examples.templates.metric",
        "params": {"src_name": "logits", "tgt_name": ("targets", "labels"), "k": 2},
    }
    cfg.val_metrics = {"top2": metric}
    cfg.probe_metrics = {"top2": metric}
    cfg.score_name = "valid/top2"
    cfg.score_factor = 1
    return cfg


def test_all_the_templates_train_together(cfg):
    Trainer = import_entrypoint(cfg.main_runner, "Trainer", field="main_runner")
    task = Trainer(0, cfg)
    task.fit()

    top2 = task.engines["evaluator"].state.metrics["valid/top2"]
    assert 0.0 <= top2 <= 1.0
    # 10 classes, top-2: chance is 0.2, and exactly 0 means it never updated.
    assert top2 > 0.0

    assert "probe/top2" in task.engines["probe"].state.metrics

    assert "train/loss_focal_avg" in task.engines["trainer"].state.metrics

    # ignite's LRScheduler spends its first call on the initial value: N epochs, N-1 decays.
    assert task.optimizer.param_groups[0]["lr"] == pytest.approx(1e-3 * 0.5 ** (cfg.max_epochs - 1))


def test_the_template_metric_reduces_rather_than_lying(cfg):
    """sync_all_reduce silently skips a name that does not exist, so check they all do.

    Single-process, so the reduction itself is not exercised.
    """
    from examples.templates.metric import TopKAccuracy

    metric = TopKAccuracy(k=2)
    reduced = metric.compute.__wrapped__ if hasattr(metric.compute, "__wrapped__") else None
    assert reduced is not None, "compute() is not wrapped: @sync_all_reduce is missing"

    metric.reset()
    for name in ("_num_correct", "_num_examples"):
        assert hasattr(metric, name), (
            f"@sync_all_reduce names {name!r} but reset() never creates it, so it would "
            "reduce nothing and the metric would stay rank-local under DDP"
        )
        assert torch.is_tensor(getattr(metric, name)), (
            f"{name} must be a tensor: sync_all_reduce cannot move a plain python float "
            "between processes"
        )


def test_the_template_dataset_only_augments_train(cfg):
    """The transform is handed to every split, so the template must not augment validation."""
    from examples.templates.augmentation import Transformation
    from examples.templates.dataset import MyDataset

    always = Transformation(p=1.0, scale=1000.0)  # unmissable if applied
    raw = MyDataset(n=8, split="valid")
    valid = MyDataset(n=8, split="valid", transform=always)
    train = MyDataset(n=8, split="train", transform=always)

    assert torch.equal(raw[0][0], valid[0][0]), "the valid split was augmented"
    assert not torch.equal(raw[0][0], train[0][0]), "the train split was NOT augmented"
