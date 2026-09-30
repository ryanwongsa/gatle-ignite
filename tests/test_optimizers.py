import pytest
import torch
import torch.nn as nn
from examples.synthetic.configs.synthetic_v0 import get_config

from gatle_ignite import BaseTrainer
from gatle_ignite.dispatch import ConfigError
from gatle_ignite.optimizers.adam import get_optimizer as adam
from gatle_ignite.optimizers.grouping import get_grouped_params
from gatle_ignite.optimizers.layer_decay import get_optimizer as layer_decay


class _Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(8, 8)
        self.norm = nn.LayerNorm(8)


class _Backbone(nn.Module):
    def __init__(self, n=3):
        super().__init__()
        self.patch_embed = nn.Linear(8, 8)
        self.blocks = nn.ModuleList(_Block() for _ in range(n))


class _Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Module()
        self.backbone.backbone = _Backbone()
        self.head = nn.Linear(8, 2)


def test_grouped_params_exclude_bias_and_norm_from_decay():
    model = _Block()
    groups = get_grouped_params(model, weight_decay=0.05)
    assert len(groups) == 2
    decay, no_decay = groups
    assert decay["weight_decay"] == 0.05
    assert no_decay["weight_decay"] == 0.0
    # biases + LayerNorm weight/bias are 1-D
    assert all(p.ndim == 1 for p in no_decay["params"])


def test_adam_optimizes_only_the_trainable_params():
    model = _Block()
    model.norm.weight.requires_grad_(False)
    optimizer = adam(model, lr=0.01, betas=(0.8, 0.9))
    assert isinstance(optimizer, torch.optim.Adam)
    params = [p for group in optimizer.param_groups for p in group["params"]]
    assert len(params) == 3  # fc.weight, fc.bias, norm.bias
    assert all(p is not model.norm.weight for p in params)
    assert optimizer.param_groups[0]["lr"] == 0.01
    assert optimizer.param_groups[0]["betas"] == (0.8, 0.9)


def test_unknown_optimizer_names_the_field_instead_of_unbound_local():
    cfg = get_config()
    cfg.optimizer_name = "gatle_ignite.optimizers.nope"
    trainer = BaseTrainer(0, cfg)
    trainer.model = _Block()
    with pytest.raises(ConfigError, match="optimizer_name"):
        trainer.build_optimizer()


def test_layer_decay_gives_earlier_layers_smaller_lrs():
    optimizer = layer_decay(_Net(), lr=1e-3, layer_decay=0.5, backbone_attr="backbone.backbone")
    lrs = sorted({g["lr"] for g in optimizer.param_groups})
    assert len(lrs) > 1
    assert max(lrs) <= 1e-3  # nothing exceeds the base lr


def test_layer_decay_rejects_a_wrong_backbone_path_loudly():
    with pytest.raises(ConfigError) as e:
        layer_decay(_Net(), backbone_attr="backbone.nope")
    assert "does not resolve" in str(e.value)
    assert "Available" in str(e.value)


def test_layer_decay_rejects_a_backbone_without_blocks():
    with pytest.raises(ConfigError, match="blocks"):
        layer_decay(_Net(), backbone_attr="backbone.backbone", blocks_attr="layers")
