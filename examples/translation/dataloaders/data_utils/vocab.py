"""The toy language. Reversed word order makes the decoder attend across the whole source.

A monotonic copy could pass by aligning frame i to token i.
"""

import torch

PAD_ID = 0
BOS_ID = 1
EOS_ID = 2
N_SPECIAL = 3

N_WORDS = 12  # source vocabulary size = target vocabulary size
TGT_VOCAB = N_SPECIAL + N_WORDS

# A fixed non-identity permutation: source word w translates to target word PERM[w].
PERM = [(5 * w + 3) % N_WORDS for w in range(N_WORDS)]

MIN_WORDS, MAX_WORDS = 2, 4
MIN_FRAMES, MAX_FRAMES = 2, 4

MAX_SRC_LEN = MAX_WORDS * MAX_FRAMES  # 16
MAX_TGT_LEN = MAX_WORDS + 2  # BOS + words + EOS = 6


def prototypes(in_dim, task_seed=1234):
    """Per-word acoustic prototypes, from their OWN generator so every split shares one task."""
    gen = torch.Generator().manual_seed(task_seed)
    return torch.randn(N_WORDS, in_dim, generator=gen)


def translate(src_words):
    """The reference translation of a list of source-word ids -> target token ids."""
    return [BOS_ID] + [N_SPECIAL + PERM[w] for w in reversed(src_words)] + [EOS_ID]
