def get_grouped_params(model, weight_decay, no_decay_on_bias_and_norm=True):
    """Biases and 1-D (norm) params skip weight decay when no_decay_on_bias_and_norm is set."""
    if not weight_decay or not no_decay_on_bias_and_norm:
        return [{"params": [p for p in model.parameters() if p.requires_grad]}]

    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim <= 1 or name.endswith(".bias"):
            no_decay.append(param)
        else:
            decay.append(param)
    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
