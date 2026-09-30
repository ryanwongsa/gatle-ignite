import torch

IMG = 32
CLASS_NAMES = ("square", "circle", "ring", "hbar", "vbar")
N_CLASSES = len(CLASS_NAMES)

_YY, _XX = torch.meshgrid(torch.arange(IMG), torch.arange(IMG), indexing="ij")


def _randint(gen, lo, hi):
    return int(torch.randint(lo, hi, (1,), generator=gen).item())


def draw(label, gen, noise=0.45):
    """One (1, IMG, IMG) image of class `label`.

    Deliberately hard: a task that saturates at 1.00 hides a broken sampler.
    """
    cy, cx = _randint(gen, 10, 23), _randint(gen, 10, 23)
    r = _randint(gen, 3, 7)
    dy, dx = (_YY - cy).abs(), (_XX - cx).abs()

    if label == 0:  # filled square
        mask = (dy <= r) & (dx <= r)
    elif label == 1:  # filled circle
        mask = (dy.float() ** 2 + dx.float() ** 2) <= float(r * r)
    elif label == 2:  # ring (hollow circle)
        d2 = dy.float() ** 2 + dx.float() ** 2
        mask = (d2 <= float(r * r)) & (d2 >= float((r - 2) * (r - 2)))
    elif label == 3:  # horizontal bar
        mask = (dy <= 2) & (dx <= r)
    elif label == 4:  # vertical bar
        mask = (dx <= 2) & (dy <= r)
    else:
        raise ValueError(f"unknown label {label}")

    img = torch.zeros(1, IMG, IMG)
    # A per-image foreground intensity, so brightness alone never identifies the class.
    img[0][mask] = 0.5 + 0.3 * torch.rand(1, generator=gen).item()
    img += noise * torch.randn(1, IMG, IMG, generator=gen)
    return img


def make_split(class_counts, seed, noise=0.45):
    """(images, labels) for a split whose class i has class_counts[i] samples."""
    gen = torch.Generator().manual_seed(seed)
    xs, ys = [], []
    for label, count in enumerate(class_counts):
        for _ in range(count):
            xs.append(draw(label, gen, noise=noise))
            ys.append(label)
    x = torch.stack(xs)
    y = torch.tensor(ys, dtype=torch.long)
    # Shuffle once, deterministically: ordered data is a bug that shows only under DDP sharding.
    perm = torch.randperm(len(y), generator=gen)
    return x[perm], y[perm]
