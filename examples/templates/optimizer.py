"""TEMPLATE: an optimizer for cfg.optimizer_name; optimizer_params are splatted in as kwargs."""

import torch


def get_optimizer(model, lr=1e-3, weight_decay=0.0, no_decay_on_norm_and_bias=True, **kwargs):
    """Return the optimizer object the trainer holds; it need not be a torch.optim.Optimizer.

    `model` is already DDP-wrapped: parameters() sees through it, but a submodule does not.
    Reach one via `getattr(model, "module", model).gen`, or it fails only on the cluster.
    The framework asks only for param_groups, state_dict, load_state_dict, zero_grad, step.
    """
    if not no_decay_on_norm_and_bias:
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay, **kwargs)

    # Biases and norm scales have too few parameters to overfit; decaying them mostly hurts.
    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim <= 1 or name.endswith(".bias"):
            no_decay.append(param)
        else:
            decay.append(param)

    return torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=lr,
        **kwargs,
    )
