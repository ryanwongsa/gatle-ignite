from torchvision import datasets, transforms

from gatle_ignite import build_dataloader

MEAN, STD = (0.1307,), (0.3081,)


def get_ds(ds_params, transform=None):
    """MNIST. Needs the [examples] extra (torchvision) and downloads on first run."""
    train = ds_params.get("split", "train") == "train"

    plain = transforms.Compose([transforms.ToTensor(), transforms.Normalize(MEAN, STD)])

    # One transform reaches every split; augmenting eval would quietly depress every metric.
    tfm = (transform or plain) if train else plain

    ds = datasets.MNIST(
        root=ds_params.get("root", "./data"), train=train, download=True, transform=tfm
    )
    return build_dataloader(ds, ds_params), {"length": len(ds), "num_classes": 10}
