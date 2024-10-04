
# from https://github.com/yolomeus/DMLab2020VFAE/blob/master/losses.py
from math import pi, sqrt

import torch
import torch.nn as nn

class FastMMD(nn.Module):
    def __init__(self, out_features, gamma):
        super().__init__()
        self.gamma = gamma
        self.out_features = out_features

    def forward(self, a, b):
        in_features = a.shape[-1]

        w_rand = torch.randn((in_features, self.out_features), device=a.device)
        b_rand = torch.zeros((self.out_features,), device=a.device).uniform_(0, 2 * pi)

        phi_a = self._phi(a, w_rand, b_rand).mean(dim=0)
        phi_b = self._phi(b, w_rand, b_rand).mean(dim=0)
        mmd = torch.norm(phi_a - phi_b, 2)

        return mmd

    def _phi(self, x, w, b):
        scale_a = sqrt(2 / self.out_features)
        scale_b = sqrt(2 / self.gamma)
        out = scale_a * (scale_b * (x @ w + b)).cos()
        return out