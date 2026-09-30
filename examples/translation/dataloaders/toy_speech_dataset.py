import torch
from torch.utils.data import Dataset

from examples.translation.dataloaders.data_utils.vocab import (
    EOS_ID,
    MAX_FRAMES,
    MAX_WORDS,
    MIN_FRAMES,
    MIN_WORDS,
    N_WORDS,
    PAD_ID,
    TGT_VOCAB,
    prototypes,
    translate,
)
from gatle_ignite import build_dataloader


class ToySpeechTranslation(Dataset):
    def __init__(self, n=2048, in_dim=32, seed=0, noise=0.15, task_seed=1234):
        self.n = n
        self.in_dim = in_dim
        self.seed = seed
        self.noise = noise
        self.proto = prototypes(in_dim, task_seed)

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        # Seeded per index, so an index is always the same sample; the split seed varies them.
        gen = torch.Generator().manual_seed(self.seed * 1_000_003 + idx)

        n_words = int(torch.randint(MIN_WORDS, MAX_WORDS + 1, (1,), generator=gen))
        words = [int(w) for w in torch.randint(0, N_WORDS, (n_words,), generator=gen)]

        frames = []
        for w in words:
            k = int(torch.randint(MIN_FRAMES, MAX_FRAMES + 1, (1,), generator=gen))
            noise = self.noise * torch.randn(k, self.in_dim, generator=gen)
            frames.append(self.proto[w].unsqueeze(0).expand(k, -1) + noise)

        src = torch.cat(frames, dim=0)  # (T, in_dim), T in [4, 16]
        tgt = torch.tensor(translate(words), dtype=torch.long)  # (U,), U in [4, 6]
        return src, tgt


def collate_fn(batch):
    """Pad (src (T,D), tgt (U,)) pairs. Module-level, not a lambda, so it pickles to workers."""
    srcs, tgts = zip(*batch)
    bs = len(batch)
    in_dim = srcs[0].shape[1]
    t_max = max(s.shape[0] for s in srcs)
    u_max = max(t.shape[0] for t in tgts)

    src_feats = torch.zeros(bs, t_max, in_dim)
    src_pad_mask = torch.ones(bs, t_max, dtype=torch.bool)  # True == padded == ignore
    tgt = torch.full((bs, u_max), PAD_ID, dtype=torch.long)

    for i, (s, t) in enumerate(zip(srcs, tgts)):
        src_feats[i, : s.shape[0]] = s
        src_pad_mask[i, : s.shape[0]] = False
        tgt[i, : t.shape[0]] = t

    # Teacher forcing: read tgt[:-1], predict tgt[1:]. Padding stays PAD_ID, which the loss
    # ignores: a padded loss trains toward <pad> and still looks converged.
    return {
        "src_feats": src_feats,
        "src_pad_mask": src_pad_mask,
        "tgt_in": tgt[:, :-1].contiguous(),
        "tgt_out": tgt[:, 1:].contiguous(),
    }


def get_ds(ds_params, transform=None):
    ds = ToySpeechTranslation(
        n=ds_params.get("n", 2048),
        in_dim=ds_params.get("in_dim", 32),
        seed=ds_params.get("seed", 0),
        noise=ds_params.get("noise", 0.15),
        task_seed=ds_params.get("task_seed", 1234),
    )
    dl = build_dataloader(ds, {**ds_params, "collate_fn": collate_fn})
    # `info` is how the WER metric learns the pad and eos ids.
    info = {
        "length": len(ds),
        "pad_id": PAD_ID,
        "eos_id": EOS_ID,
        "tgt_vocab": TGT_VOCAB,
    }
    return dl, info
