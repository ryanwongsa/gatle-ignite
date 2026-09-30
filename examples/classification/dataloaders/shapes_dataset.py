from torch.utils.data import Dataset

from examples.classification.dataloaders.data_utils.shapes import CLASS_NAMES, N_CLASSES, make_split
from gatle_ignite import build_dataloader

# The imbalance the sampler exists to undo: 20x between the head and the tail class.
TRAIN_COUNTS = (600, 300, 150, 60, 30)
# Valid is BALANCED: it is what picks the checkpoint, and every class should count equally.
EVAL_COUNTS = (100, 100, 100, 100, 100)
# Test mirrors the imbalanced deployment distribution, so test/acc and test/bal_acc disagree.
DEPLOY_COUNTS = (400, 200, 100, 40, 20)


class ShapesDataset(Dataset):
    def __init__(self, split="train", class_counts=TRAIN_COUNTS, seed=0, transform=None):
        self.split = split
        self.transform = transform
        self.x, self.y = make_split(class_counts, seed=seed)

    def __len__(self):
        return len(self.y)

    @property
    def labels(self):
        """Exposed for the sampler, which weights by class frequency."""
        return self.y

    def __getitem__(self, idx):
        x, y = self.x[idx], self.y[idx]
        # One transform reaches every split: augmenting valid/test would skew checkpoint choice.
        if self.transform is not None and self.split == "train":
            x = self.transform(x)
        return x, y


def get_ds(ds_params, transform=None):
    split = ds_params.get("split", "train")
    counts = tuple(ds_params.get("class_counts", TRAIN_COUNTS if split == "train" else EVAL_COUNTS))
    ds = ShapesDataset(
        split=split,
        class_counts=counts,
        seed=ds_params.get("seed", 0),
        transform=transform,
    )
    info = {
        "length": len(ds),
        "num_classes": N_CLASSES,
        "class_names": list(CLASS_NAMES),
        "class_counts": list(counts),
    }
    return build_dataloader(ds, ds_params), info
