# 6. Optimizer and schedule

`optimizer/my_optimizer.py` → `cfg.optimizer_name` · `scheduler/my_schedule.py` → `cfg.lr_scheduler`

```python
def get_optimizer(model, **params)   -> the object the trainer holds
def get_scheduler(cfg, train_dl, optimizer, engines) -> (scheduler, engine_name, event_name)
```

```python
cfg.optimizer_name   = "gatle_ignite.optimizers.adamw"
cfg.optimizer_params = {"lr": 1e-3, "weight_decay": 0.01}
cfg.lr_scheduler     = "gatle_ignite.schedulers.warmup_cosine"
cfg.lr_scheduler_params = {"warmup_epochs": 1}
```

Builtins, always named in full: `gatle_ignite.optimizers.{adam,adamw,sgd,layer_decay}` and
`gatle_ignite.schedulers.{cosine,warmup_cosine,step,plateau}`. The LR schedule reads `lr` from
`optimizer_params`.

`engine_name` may be any key of `engines`, so a plateau schedule can watch an engine your task
declared rather than only the built-in evaluator. `cfg.lr_scheduler = None` means no scheduling.

!!! tip "`get_optimizer` need not return a `torch.optim.Optimizer`"
    The trainer only asks for `param_groups`, `state_dict` and `load_state_dict`, plus
    `zero_grad` and `step` if you keep the default `train_step`. One object holding two real
    optimizers is how you train a GAN; there is no second optimizer field.

**Templates:** [optimizer](../templates.md#optimizer) · [scheduler](../templates.md#scheduler)
