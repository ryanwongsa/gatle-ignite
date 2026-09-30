"""A real research task's customisations, each asserted to have done something."""

import pytest
import torch
from examples.synthetic.configs.synthetic_v0 import get_config
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from gatle_ignite import BaseTrainer, EngineSpec, to_device
from gatle_ignite.trainer.specs import EVAL_SPECS

IN_DIM, N_CLASSES, HIDDEN = 64, 10, 32


class ProtoModel(nn.Module):
    """Carries prototype state the trainer mutates between engine runs."""

    def __init__(self):
        super().__init__()
        self.backbone = nn.Linear(IN_DIM, HIDDEN)
        self.head = nn.Linear(HIDDEN, N_CLASSES)
        self.register_buffer("prototypes", torch.zeros(N_CLASSES, HIDDEN))
        self.register_buffer("counts", torch.zeros(N_CLASSES))

    def forward(self, x, use_prototypes=False):
        feat = self.backbone(x)
        out = {"features": feat, "logits": self.head(feat)}
        if use_prototypes:
            protos = nn.functional.normalize(self.prototypes, dim=1)
            out["logits"] = nn.functional.normalize(feat, dim=1) @ protos.T
        return out

    @torch.no_grad()
    def add_prototypes(self, features, ids):
        self.prototypes.index_add_(0, ids, features.float())
        self.counts.index_add_(0, ids, torch.ones_like(ids, dtype=torch.float))

    @torch.no_grad()
    def finalize(self):
        self.prototypes /= self.counts.clamp(min=1).unsqueeze(1)

    @torch.no_grad()
    def reset_prototypes(self):
        self.prototypes.zero_()
        self.counts.zero_()


def _task_data(n, seed):
    """A learnable task: labels are a fixed linear function of the inputs."""
    wg = torch.Generator().manual_seed(1234)
    weight = torch.randn(IN_DIM, N_CLASSES, generator=wg)
    xg = torch.Generator().manual_seed(seed)
    x = torch.randn(n, IN_DIM, generator=xg)
    return x, (x @ weight).argmax(dim=1)


class ProbeTrainer(BaseTrainer):
    grad_scaled = []

    def prep_batch(self, batch, split="train", **kwargs):
        x, y = batch
        key = "gloss_id" if split == "dictionary" else "labels"
        return to_device({"model_input": {"x": x}, "targets": {key: y}})

    def build_model(self):
        return ProtoModel()

    # --- an extra dataloader the framework knows nothing about ---
    def build_dataloaders(self):
        dls, infos = super().build_dataloaders()
        x, y = _task_data(64, seed=7)
        self.dict_dl = DataLoader(TensorDataset(x, y), batch_size=16)
        return dls, infos

    # --- an engine scoring against prototypes, declared in one line ---
    def eval_specs(self):
        return EVAL_SPECS + (EngineSpec.for_split("proto", ds_prefix="valid"),)

    def eval_step(self, engine, batch, split="valid"):
        if split != "proto":
            return super().eval_step(engine, batch, split=split)
        self.model.eval()
        x = self.prep_batch(batch, split=split)
        with torch.no_grad():
            y_pred = self.model(**x["model_input"], use_prototypes=True)
        return {"y_pred": y_pred, "target": x}

    # --- per-name gradient rescaling, in place of a manual-gradient hook ---
    def backward(self, loss, step=True):
        loss.backward()
        for name, param in self.model.named_parameters():
            if param.grad is not None and "backbone" in name:
                param.grad *= 0.1
                self.grad_scaled.append(name)
        if step:
            self.optimizer.step()

    # --- the extractor mutates model state before the proto engine runs ---
    def extract_prototypes(self):
        self.model.eval()
        self.model.reset_prototypes()
        for batch in self.dict_dl:
            x = self.prep_batch(batch, split="dictionary")
            with torch.no_grad():
                y_pred = self.model(**x["model_input"])
            self.model.add_prototypes(y_pred["features"], x["targets"]["gloss_id"])
        self.model.finalize()

    def attach_runner(self):
        from ignite.engine import Events

        super().attach_runner()
        trainer, proto = self.engines["trainer"], self.engines["proto"]

        @trainer.on(Events.EPOCH_STARTED)
        def _extract(_e):
            self.extract_prototypes()

        @proto.on(Events.EPOCH_COMPLETED)
        def _inject(engine):
            engine.state.metrics["proto/spread"] = self.model.prototypes.std().item()


@pytest.fixture
def cfg(tmp_path):
    cfg = get_config()
    cfg.save_dir = str(tmp_path)
    cfg.max_epochs = 2
    cfg.logger_name = []
    cfg.main_runner = "tests.test_extension_surface"
    cfg.train_ds_params["n"] = 512
    cfg.valid_ds_params["n"] = 128
    cfg.proto_metrics = {
        "acc": {
            "cls_name": "examples.synthetic.metrics.accuracy",
            "params": {"src_name": "logits", "tgt_name": ("targets", "labels")},
        }
    }
    cfg.every_proto = 1
    return cfg


def test_a_real_task_worth_of_customisation_works(cfg):
    ProbeTrainer.grad_scaled = []
    task = ProbeTrainer(0, cfg)
    task.setup()
    before = task.model.prototypes.clone()
    task.fit()

    assert len(task.dict_dl.dataset) == 64

    assert not torch.allclose(before, task.model.prototypes), "prototypes never changed"
    assert task.counts_ok(), "the extractor did not see every class"

    acc = task.engines["proto"].state.metrics["proto/acc"]
    assert acc > 0.15, f"prototype accuracy {acc} is chance: the extractor did nothing useful"

    assert "proto/spread" in task.engines["proto"].state.metrics

    assert any("backbone" in n for n in ProbeTrainer.grad_scaled)


def test_gradient_rescaling_hits_only_the_named_parameters(cfg):
    task = ProbeTrainer(0, cfg)
    task.setup()
    batch = next(iter(task.dls["train"]))
    x = task.prep_batch(batch)
    task.optimizer.zero_grad(set_to_none=True)
    y = task.model(**x["model_input"])
    loss, _ = task.loss_fn(y, x)
    loss.backward()

    raw = {n: p.grad.norm().item() for n, p in task.model.named_parameters()}
    for name, param in task.model.named_parameters():
        if param.grad is not None and "backbone" in name:
            param.grad *= 0.1
    scaled = {n: p.grad.norm().item() for n, p in task.model.named_parameters()}

    for name in raw:
        expected = raw[name] * (0.1 if "backbone" in name else 1.0)
        assert scaled[name] == pytest.approx(expected, rel=1e-5), name


def _counts_ok(self):
    return int(self.model.counts.sum().item()) == 64


ProbeTrainer.counts_ok = _counts_ok
Trainer = ProbeTrainer  # cfg.main_runner points here
