"""A collate_fn chosen by dotted path. It returns a dict because prep_batch maps any shape."""

import torch


def get_collate_fn(tag=0):
    """The entrypoint the framework calls. Return the callable torch hands a list."""

    def collate(batch):
        xs = torch.stack([item[0] for item in batch])
        ys = torch.stack([item[1] for item in batch])
        return {"x": xs, "y": ys, "tag": tag}

    return collate
