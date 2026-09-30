"""The encoder features "h1"/"h2" feed the probe; NT-Xent compares the projections "z1"/"z2".

`x2=None` lets the eval engine pass view 1 alone and skip the second pass.
"""

import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, in_dim=64, hidden=256, feat_dim=128, proj_dim=32):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Linear(hidden, feat_dim),
        )
        self.projector = nn.Sequential(
            nn.Linear(feat_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, proj_dim),
        )

    def forward(self, x1, x2=None):
        h1 = self.encoder(x1)
        out = {"h1": h1, "z1": F.normalize(self.projector(h1), dim=-1)}
        if x2 is not None:
            h2 = self.encoder(x2)
            out["h2"] = h2
            out["z2"] = F.normalize(self.projector(h2), dim=-1)
        return out
