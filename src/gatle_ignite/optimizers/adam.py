"""Adam. No weight-decay grouping: use adamw if you want decay."""

import ignite.distributed as idist
import torch


def get_optimizer(model, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0, **kwargs):
    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr,
        betas=tuple(betas),
        eps=eps,
        weight_decay=weight_decay,
        **kwargs,
    )
    return idist.auto_optim(optimizer)
