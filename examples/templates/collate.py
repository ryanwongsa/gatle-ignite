"""TEMPLATE: a collate fn for *_ds_params.collate_fn.cls_name; params go to get_collate_fn.

ds_params["collate_fn"] also takes a plain callable; the dotted form lets a config choose.
"""

import torch


def get_collate_fn(pad_id=0):
    """Return the callable torch hands a list of samples.

    It must be picklable for num_workers > 0: a closure here is, a lambda in your get_ds is not.
    """

    def collate(batch):
        # Any shape will do: prep_batch maps whatever this returns into model_input/targets.
        xs, ys = zip(*batch)
        lengths = torch.tensor([len(x) for x in xs])
        padded = torch.nn.utils.rnn.pad_sequence(xs, batch_first=True, padding_value=pad_id)
        return {"x": padded, "lengths": lengths, "y": torch.stack(ys)}

    return collate
