"""Dataset and augmentation dispatch, DDP-aware dataloaders, and the exact eval sampler."""

import warnings

import ignite.distributed as idist
import torch
from torch.utils.data import Sampler

from gatle_ignite.dispatch import ConfigError, import_entrypoint


class ExactDistributedSampler(Sampler):
    """Shard a dataset across ranks with no padding and no dropping.

    Safe for evaluation because ignite's metrics sum counts across ranks. NOT safe for
    training: uneven per-rank batch counts deadlock the all-reduce.
    """

    def __init__(self, dataset, num_replicas, rank):
        if not 0 <= rank < num_replicas:
            raise ValueError(f"rank {rank} out of range for num_replicas {num_replicas}")
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        # Contiguous blocks, not a stride: LMDB and webdataset-style stores read runs faster.
        total = len(dataset)
        base, extra = divmod(total, num_replicas)
        start = rank * base + min(rank, extra)
        self.indices = list(range(start, start + base + (1 if rank < extra else 0)))

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)

    def set_epoch(self, epoch):
        """No-op: the order is fixed, which is what evaluation wants."""


def build_dataloader(dataset, ds_params, sampler=None):
    """Build a DDP-aware dataloader from a dataset and the config's ds_params.

    Recognised keys: bs (the total across GPUs on every split), num_workers, shuffle, drop_last,
    pin_memory, collate_fn, sampler_params, exact_sharding; anything else is the dataset's own
    business. sampler_params and collate_fn are dotted: get_sampler(dataset, **params) and
    get_collate_fn(**params). BaseTrainer sets exact_sharding on valid and test itself.
    """
    if sampler is None:
        sampler = _build_sampler(dataset, ds_params.get("sampler_params", None))

    if ds_params.get("exact_sharding", False) and idist.get_world_size() > 1:
        if sampler is not None:
            # We cannot tell whether a custom sampler selects or orders, so keep it and warn.
            warnings.warn(
                "exact_sharding is set but this split has a custom sampler, so under DDP padded "
                "duplicates will be scored. Remove the sampler to shard this split exactly.",
                stacklevel=2,
            )
        else:
            return _build_exact_sharded_loader(dataset, ds_params)

    batch_size = ds_params.get("bs", 1)
    drop_last = ds_params.get("drop_last", False)

    kwargs = {
        "batch_size": batch_size,
        "num_workers": ds_params.get("num_workers", 0),
        "drop_last": drop_last,
        # Pinning only helps for host->device copies; on CPU it is a no-op that warns.
        "pin_memory": ds_params.get("pin_memory", torch.cuda.is_available()),
    }

    if sampler is not None:
        # torch rejects sampler+shuffle, and shuffle dropping out silently deserves a warning.
        if ds_params.get("shuffle", False):
            warnings.warn(
                "ds_params sets both 'shuffle': True and a sampler, so shuffle is ignored and "
                "the sampler decides the order. Remove 'shuffle' to make that explicit.",
                stacklevel=2,
            )
        kwargs["sampler"] = sampler
    else:
        kwargs["shuffle"] = ds_params.get("shuffle", False)

    collate_fn = resolve_collate_fn(ds_params)
    if collate_fn is not None:
        kwargs["collate_fn"] = collate_fn

    if drop_last and len(dataset) < batch_size:
        # An empty loader trains on nothing and still reports success.
        warnings.warn(
            f"drop_last=True with {len(dataset)} sample(s) and bs={batch_size} yields no batches, "
            f"so this split is silently skipped. Set drop_last=False or lower bs.",
            stacklevel=2,
        )

    return idist.auto_dataloader(dataset, **kwargs)


def _build_exact_sharded_loader(dataset, ds_params):
    """Not idist.auto_dataloader: it wraps any sampler in a DistributedProxySampler, which pads."""
    from torch.utils.data import DataLoader

    batch_size = ds_params.get("bs", 1)
    num_workers = ds_params.get("num_workers", 0)
    world_size = idist.get_world_size()
    if world_size > 1:
        # bs is the total across ranks on every split: split it by auto_dataloader's own rules.
        if batch_size >= world_size:
            batch_size //= world_size
        nproc = idist.get_nproc_per_node()
        if num_workers >= nproc:
            num_workers = (num_workers + nproc - 1) // nproc

    kwargs = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": ds_params.get("pin_memory", torch.cuda.is_available()),
        # Never drop on an eval split: a dropped tail is a silently smaller test set.
        "drop_last": False,
        "sampler": ExactDistributedSampler(
            dataset, num_replicas=idist.get_world_size(), rank=idist.get_rank()
        ),
    }
    collate_fn = resolve_collate_fn(ds_params)
    if collate_fn is not None:
        kwargs["collate_fn"] = collate_fn
    return DataLoader(dataset, **kwargs)


def _build_sampler(dataset, sampler_params):
    if not sampler_params:
        return None
    get_sampler = import_entrypoint(
        sampler_params["cls_name"], "get_sampler", field="sampler_params.cls_name"
    )
    return get_sampler(dataset, **sampler_params.get("params", {}))


def resolve_collate_fn(ds_params):
    """The collate for this split: a live callable, or a dotted path. None means torch's."""
    collate = ds_params.get("collate_fn", None)
    if collate is None or callable(collate):
        return collate
    if not _is_dispatch(collate):
        raise ConfigError(
            f"ds_params['collate_fn'] must be a callable or "
            f"{{'cls_name': 'dotted.path', 'params': {{...}}}}, got {collate!r}"
        )
    get_collate_fn = import_entrypoint(
        collate["cls_name"], "get_collate_fn", field="collate_fn.cls_name"
    )
    return get_collate_fn(**collate.get("params", {}))


def _is_dispatch(value):
    return hasattr(value, "get") and value.get("cls_name", None)


def get_dataset(ds_name, ds_params, transform=None, field="train_ds_name"):
    """-> (dataloader, info). `info["length"]` sizes the metrics and progress handlers."""
    get_ds = import_entrypoint(ds_name, "get_ds", field=field)
    result = get_ds(ds_params or {}, transform=transform)

    if not isinstance(result, tuple) or len(result) != 2:
        raise ConfigError(
            f"{ds_name}.get_ds must return (dataloader, info_dict), got {type(result)}"
        )
    dataloader, info = result
    if not hasattr(info, "get") or "length" not in info:
        raise ConfigError(f"{ds_name}.get_ds info dict must contain 'length'")
    return dataloader, info


def get_aug(aug_name, aug_params):
    if not aug_name:
        return None
    return import_entrypoint(aug_name, "Transformation", field="aug_name")(**(aug_params or {}))
