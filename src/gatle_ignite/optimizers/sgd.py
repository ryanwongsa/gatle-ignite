"""SGD with momentum, Nesterov by default."""

import ignite.distributed as idist
import torch

from gatle_ignite.optimizers.grouping import get_grouped_params


def get_optimizer(
    model,
    lr=0.1,
    momentum=0.9,
    weight_decay=0.0,
    nesterov=True,
    no_decay_on_bias_and_norm=True,
    **kwargs,
):
    params = get_grouped_params(model, weight_decay, no_decay_on_bias_and_norm)
    optimizer = torch.optim.SGD(
        params,
        lr=lr,
        momentum=momentum,
        weight_decay=weight_decay,
        nesterov=nesterov,
        **kwargs,
    )
    return idist.auto_optim(optimizer)
