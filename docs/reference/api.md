# Python API

The public surface: what `from gatle_ignite import ...` gives you. Everything else is internal and
may move.

Most tasks need only `BaseTrainer`, `base_config` and `to_device`.

::: gatle_ignite.BaseTrainer
    options:
      members:
        - prep_batch
        - forward
        - build_model
        - dict_metric_from_list
        - setup
        - fit
        - evaluate
        - eval_specs
        - train_spec
        - eval_context
        - run_eval_engine
        - backward
        - extra_to_save
        - build_dataloaders

::: gatle_ignite.base_config

::: gatle_ignite.build_dataloader

::: gatle_ignite.to_device

::: gatle_ignite.get_value

::: gatle_ignite.ConfigError
