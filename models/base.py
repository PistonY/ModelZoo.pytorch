# -*- coding: utf-8 -*-
# @Author  : DevinYang(pistonyang@gmail.com)
__all__ = ['SELayer']

from torch import nn


class SELayer(nn.Module):
    def __init__(self, in_c, reducation_c, act=nn.ReLU(True)):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.lin1 = nn.Linear(in_c, reducation_c)
        self.act = act
        self.lin2 = nn.Linear(reducation_c, in_c)

    def forward(self, x):
        out = self.avg_pool(x)
        out = out.view(out.size(0), -1)
        out = self.lin1(out)
        out = self.act(out)
        out = self.lin2(out)
        out = x * out
        return out
