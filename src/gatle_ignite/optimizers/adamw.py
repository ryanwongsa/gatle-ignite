"""AdamW, with bias and norm params excluded from weight decay."""

import ignite.distributed as idist
import torch

from gatle_ignite.optimizers.grouping import get_grouped_params


def get_optimizer(
    model,
    lr=1e-3,
    weight_decay=0.01,
    betas=(0.9, 0.999),
    eps=1e-8,
    no_decay_on_bias_and_norm=True,
    **kwargs,
):
    params = get_grouped_params(model, weight_decay, no_decay_on_bias_and_norm)
    optimizer = torch.optim.AdamW(
        params, lr=lr, betas=tuple(betas), eps=eps, weight_decay=weight_decay, **kwargs
    )
    return idist.auto_optim(optimizer)
