# coding=utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    def __init__(self, d, p=-1.0, eps=1e-8, bias=False):
        """
            Root Mean Square Layer Normalization
        :param d: model size
        :param p: partial RMSNorm, valid value (0, 1). Values < 0 or >= 1
            use the full hidden dimension. Default -1.0 disables partial RMS.
        :param eps:  epsilon value, default 1e-8
        :param bias: whether use bias term for RMSNorm, disabled by
            default because RMSNorm doesn't enforce re-centering invariance.
        """
        super(RMSNorm, self).__init__()

        if d <= 0:
            raise ValueError("d must be a positive integer")
        if p == 0.0:
            raise ValueError("p must be in (0, 1), or negative to disable pRMSNorm")

        self.eps = eps
        self.d = d
        self.p = p
        self.bias = bias

        self.scale = nn.Parameter(torch.ones(d))

        if self.bias:
            self.offset = nn.Parameter(torch.zeros(d))

    def forward(self, x):
        if x.size(-1) != self.d:
            raise ValueError(
                "RMSNorm expected last dimension {}, got {}".format(
                    self.d, x.size(-1)
                )
            )

        if self.p < 0.0 or self.p >= 1.0:
            mean_square = torch.mean(x.pow(2), dim=-1, keepdim=True)
        else:
            partial_size = int(self.d * self.p)
            partial_size = max(1, partial_size)
            partial_x = x[..., :partial_size]
            mean_square = torch.mean(partial_x.pow(2), dim=-1, keepdim=True)

        x_normed = x * torch.rsqrt(mean_square + self.eps)

        if self.bias:
            return self.scale * x_normed + self.offset

        return self.scale * x_normed

    def extra_repr(self):
        return "d={}, p={}, eps={}, bias={}".format(
            self.d, self.p, self.eps, self.bias
        )
