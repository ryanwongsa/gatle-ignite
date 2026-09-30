"""The config schema. Fields a project adds for itself are ignored by the framework."""

from ml_collections import config_dict

REQUIRED_FIELDS = (
    "name",
    "save_dir",
    "main_runner",
    "model_name",
    "train_ds_name",
    "criterion_name",
    "optimizer_name",
)

# Not pre-set by base_config(), because their presence carries the meaning.
OPTIONAL_KNOWN = {
    "valid_ds_name": "Dotted path to the validation dataset module. Absent -> no evaluator engine.",
    "valid_ds_params": "Params passed to the validation `get_ds`.",
    "test_ds_name": "Dotted path to the test dataset module. Absent -> no tester engine.",
    "test_ds_params": "Params passed to the test `get_ds`.",
    "run": "Presence puts the trainer in inference mode: load a checkpoint and evaluate, never train.",
    "load_from_ckpt": 'Which checkpoint inference mode loads: `"best"` (default) or `"latest"`.',
    "wandb_entity": "W&B entity to log under. Absent -> your default entity.",
    "discord_url": "Discord webhook URL. Prefer the `DISCORD_WEBHOOK_URL` env var: a webhook is a credential.",
}


def base_config():
    """Return a ConfigDict with every optional field already set.

    A task's `get_config()` starts here and overrides only what it changes. Fields the
    framework knows but does not pre-set live in `OPTIONAL_KNOWN` above.
    """
    cfg = config_dict.ConfigDict()

    # --- identity / wiring ---
    cfg.name = config_dict.placeholder(
        str
    )  # Identifies the run: the W&B id, and the default save_dir stem.
    cfg.save_dir = config_dict.placeholder(str)  # Where checkpoints go. Resume reads from here too.
    cfg.project_name = "gatle"  # Groups runs in W&B. Not used by the text logger.
    cfg.main_runner = config_dict.placeholder(
        str
    )  # Dotted path to the module exposing your `Trainer`.
    cfg.seed = 42  # Seeds python, numpy and torch before anything is built.

    # --- model ---
    cfg.model_name = config_dict.placeholder(
        str
    )  # Dotted path to a module exposing `Model(**model_params)`.
    cfg.model_params = {}  # Splatted into `Model(...)`.
    cfg.auto_model_params = {
        "find_unused_parameters": False,
        "sync_bn": True,
    }  # kwargs for idist.auto_model (DDP wrapping). sync_bn needs a GPU backend.

    # --- data ---
    cfg.train_ds_name = config_dict.placeholder(
        str
    )  # Dotted path to a module exposing `get_ds(ds_params, transform)`.
    cfg.train_ds_params = {}  # Passed to your `get_ds`. Batch size lives here (`bs`), not at top level.
    cfg.aug_name = config_dict.placeholder(
        str
    )  # Optional. Built once and passed to EVERY split; the dataset decides who gets it.
    cfg.aug_params = {}  # Splatted into `Transformation(...)`.

    # --- loss ---
    cfg.criterion_name = config_dict.placeholder(
        str
    )  # Usually `gatle_ignite.losses.composite`, even for a single term.
    cfg.criterion_params = {}  # For the composite: `{'dict_of_loss_params': {...}}`.

    # --- optimizer / scheduler ---
    cfg.optimizer_name = config_dict.placeholder(
        str
    )  # Dotted path. Builtins are named in full: `gatle_ignite.optimizers.adamw`.
    cfg.optimizer_params = {}  # Splatted into `get_optimizer(model, ...)`. The LR schedule reads `lr` from here.
    cfg.lr_scheduler = config_dict.placeholder(str)  # Dotted path. None = no scheduling.
    cfg.lr_scheduler_params = {}  # Splatted into `get_scheduler(...)`.
    cfg.grad_clip_norm = config_dict.placeholder(float)  # Clip gradients by total norm. None = off.
    cfg.grad_clip_value = config_dict.placeholder(float)  # Clip gradients elementwise. None = off.
    cfg.accum_steps = 1  # Batches per optimizer step. N emulates N*bs at the memory of bs. Clipping applies to the accumulated gradient.

    # --- loop ---
    cfg.max_epochs = (
        1  # Also sets the LR schedule's geometry, so changing it on resume replays the old curve.
    )
    cfg.every_val = (
        1  # Run the evaluator every N epochs. 0 disables validation, mirroring every_test.
    )
    cfg.every_test = 0  # 0 disables the tester engine entirely
    cfg.train_length = config_dict.placeholder(
        int
    )  # Cap an epoch to N iterations. None = the full dataloader. Handy for smoke runs.
    cfg.early_stop_patience = (
        0  # Stop after N evaluations with no improvement in score_name. 0 = never stop early.
    )
    cfg.early_stop_after = (
        0  # Epochs to train before the patience counter arms. 0 = arm immediately.
    )
    cfg.val_length = config_dict.placeholder(int)  # As train_length, for the evaluator.
    cfg.test_length = config_dict.placeholder(int)  # As train_length, for the tester.

    # --- precision ---
    cfg.amp_dtype = "auto"  # auto (bf16 where supported, else fp32) | bf16 | fp16 | fp32
    cfg.cudnn_benchmark = (
        False  # Autotunes per input shape. A pessimisation when shapes vary; off by default.
    )
    cfg.compile = config_dict.placeholder(
        object
    )  # torch.compile the model: True | False | "auto" (on where it is usable). Off by default; the first step pays a one-off warm-up.
    cfg.compile_params = {}  # Splatted into nn.Module.compile(): mode, dynamic, backend, fullgraph.

    # --- metrics / scoring ---
    cfg.train_metrics = {}  # {"acc": {"cls_name": "...", "params": {...}}}. Loss averages are added automatically.
    cfg.val_metrics = {}  # As train_metrics, on the evaluator. Keys are namespaced "valid/<name>".
    cfg.tester_metrics = {}  # As train_metrics, on the tester. Keys are namespaced "test/<name>".
    cfg.score_name = config_dict.placeholder(
        str
    )  # Metric that selects the best checkpoint, e.g. "valid/acc". None = latest only.
    cfg.score_factor = 1  # Checkpointing keeps the MAXIMUM, so use -1 for loss/CER/WER.
    cfg.tester_score_name = config_dict.placeholder(
        str
    )  # As score_name, for the test engine -> `test_best_result*`.
    cfg.tester_score_factor = 1  # As score_factor, for the test engine.

    # --- checkpointing ---
    cfg.save_ckpt = True  # Write checkpoints at all.
    cfg.n_saved = (
        1  # How many of each kind to keep, >= 1 or None for all. Above 1, `best` is by score.
    )
    cfg.resume = False  # Continue from the newest `latest_epoch*` in save_dir. Safe to leave on; warns if there is none.
    cfg.model_checkpoint_dir = (
        ""  # Load WEIGHTS ONLY from this file (fine-tuning, or a clean warm start).
    )
    cfg.strict = True  # strict= for the model_checkpoint_dir load.

    # --- logging ---
    cfg.logger_name = [
        "text"
    ]  # Sinks to enable. The one field taking short names: text, pbar, wandb, discord.
    cfg.log_every = 100  # Iterations between LR / grad-norm points.
    cfg.watch_grad = False  # Log gradient norms. Costs a pass over the parameters.
    cfg.tags = []  # Passed to W&B.

    # --- distributed ---
    cfg.dist_backend = (
        "nccl"  # Only applies with >1 process; below that a run is single-process with no backend.
    )
    cfg.nnodes = 1  # Number of machines. >1 needs the same command, and node_rank, on every node.
    cfg.node_rank = 0  # This machine's index, 0..nnodes-1. Node 0 is where master_addr must point.
    cfg.nproc_per_node = config_dict.placeholder(
        int
    )  # Processes per machine. None = one per visible GPU (or 1 on CPU).
    cfg.master_addr = config_dict.placeholder(
        str
    )  # Rendezvous host. None = MASTER_ADDR, else 127.0.0.1. Required for nnodes > 1.
    cfg.master_port = config_dict.placeholder(
        int
    )  # Rendezvous port. None = MASTER_PORT, else random on one node. Required for nnodes > 1.

    return cfg


def known_fields():
    return sorted(set(base_config().keys()) | set(OPTIONAL_KNOWN))
