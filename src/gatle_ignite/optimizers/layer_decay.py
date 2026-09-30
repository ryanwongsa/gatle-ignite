"""Layer-wise LR decay (BEiT/ViT-style), for fine-tuning a pretrained backbone.

The backbone path is explicit, not probed for, so a wrong one errors instead of mis-grouping.
"""

import ignite.distributed as idist
import torch

from gatle_ignite.dispatch import ConfigError


def _resolve(model, dotted_attr):
    module = model.module if hasattr(model, "module") else model
    if not dotted_attr:
        return module
    for part in dotted_attr.split("."):
        if not hasattr(module, part):
            raise ConfigError(
                f"backbone_attr {dotted_attr!r} does not resolve: "
                f"{type(module).__name__} has no attribute {part!r}.\n"
                f"  Available: {sorted(n for n, _ in module.named_children())}"
            )
        module = getattr(module, part)
    return module


def _layer_id(name, n_layers, blocks_attr):
    if any(k in name for k in ("cls_token", "mask_token", "pos_embed", "patch_embed")):
        return 0
    if f"{blocks_attr}." in name:
        after = name.split(f"{blocks_attr}.")[1]
        return int(after.split(".")[0]) + 1
    return n_layers


def get_optimizer(
    model,
    lr=1e-4,
    layer_decay=0.75,
    weight_decay=0.05,
    backbone_attr="",
    blocks_attr="blocks",
    betas=(0.9, 0.999),
    **kwargs,
):
    backbone = _resolve(model, backbone_attr)
    blocks = getattr(backbone, blocks_attr, None)
    if blocks is None:
        raise ConfigError(
            f"blocks_attr {blocks_attr!r} not found on {type(backbone).__name__}.\n"
            f"  Set backbone_attr to the module holding the blocks, and blocks_attr to their name."
        )

    n_layers = len(blocks) + 1
    scales = [layer_decay ** (n_layers - i) for i in range(n_layers + 1)]

    # no_weight_decay() names are backbone-relative ("pos_embed") and named_parameters()
    # model-relative, so compare on the suffix: a literal match never fires.
    no_decay = set(getattr(backbone, "no_weight_decay", lambda: set())())

    def is_no_decay(name):
        return any(name == n or name.endswith(f".{n}") for n in no_decay)

    groups = {}
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        decay = (
            0.0
            if (param.ndim == 1 or name.endswith(".bias") or is_no_decay(name))
            else weight_decay
        )
        depth = _layer_id(name, n_layers, blocks_attr)
        key = f"layer_{depth}_{'no_decay' if decay == 0.0 else 'decay'}"
        if key not in groups:
            groups[key] = {
                "lr": lr * scales[depth],
                "weight_decay": decay,
                "params": [],
            }
        groups[key]["params"].append(param)

    optimizer = torch.optim.AdamW(list(groups.values()), lr=lr, betas=tuple(betas), **kwargs)
    return idist.auto_optim(optimizer)
