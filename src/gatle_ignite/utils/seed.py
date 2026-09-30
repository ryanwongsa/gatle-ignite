import os
import random

import numpy as np
import torch


def setup_seed(seed, cudnn_benchmark=False):
    """Seed every RNG the training loop touches."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = cudnn_benchmark
