# -*- coding: utf-8 -*-
# @Author  : DevinYang(pistonyang@gmail.com)
"""Paper `PROXYLESSNAS: DIRECT NEURAL ARCHITECTURE
SEARCH ON TARGET TASK AND HARDWARE`,
 `https://arxiv.org/pdf/1812.00332.pdf`"""

__all__ = ['ProxylessGPU', 'ProxylessCPU', 'ProxylessMobile']

from .fairnet import InvertedResidual
from torch import nn


class ProxylessGPU(nn.Module):
    def __init__(self, num_classes=1000, small_input=False):
        super(ProxylessGPU, self).__init__()
        self.first_block = nn.Sequential(
            nn.Conv2d(3, 40, 3, 2 if not small_input else 1, 1, bias=False),
            nn.BatchNorm2d(40),
            nn.ReLU6(inplace=True),
            nn.Conv2d(40, 40, 3, 1, 1, groups=40, bias=False),
            nn.BatchNorm2d(40),
            nn.ReLU6(inplace=True),
            nn.Conv2d(40, 24, 1, 1, bias=False),
            nn.BatchNorm2d(24)
        )
        self.mb_blocks = nn.Sequential(
            InvertedResidual(24, 3, 32, 5, 2),
            InvertedResidual(32, 3, 56, 7, 2),
            InvertedResidual(56, 3, 56, 3, 1),
            InvertedResidual(56, 6, 112, 7, 2),
            InvertedResidual(112, 3, 112, 5, 1),
            InvertedResidual(112, 6, 128, 5, 1),
            InvertedResidual(128, 3, 128, 3, 1),
            InvertedResidual(128, 3, 128, 5, 1),
            InvertedResidual(128, 6, 256, 7, 2),
            InvertedResidual(256, 6, 256, 7, 1),
            InvertedResidual(256, 6, 256, 7, 1),
            InvertedResidual(256, 6, 256, 5, 1),
            InvertedResidual(256, 6, 432, 7, 1),
        )
        self.last_block = nn.Sequential(
            nn.Conv2d(432, 1728, 1, 1, bias=False),
            nn.BatchNorm2d(1728),
            nn.ReLU6(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )
        self.output = nn.Linear(1728, num_classes)

    def forward(self, x):
        x = self.first_block(x)
        x = self.mb_blocks(x)
        x = self.last_block(x)
        x = self.output(x)
        return x


class ProxylessCPU(nn.Module):
    def __init__(self, num_classes=1000, small_input=False):
        super(ProxylessCPU, self).__init__()
        self.first_block = nn.Sequential(
            nn.Conv2d(3, 40, 3, 2 if not small_input else 1, 1, bias=False),
            nn.BatchNorm2d(40),
            nn.ReLU6(inplace=True),
            nn.Conv2d(40, 40, 3, 1, 1, groups=40, bias=False),
            nn.BatchNorm2d(40),
            nn.ReLU6(inplace=True),
            nn.Conv2d(40, 24, 1, 1, bias=False),
            nn.BatchNorm2d(24)
        )
        self.mb_blocks = nn.Sequential(
            InvertedResidual(24, 6, 32, 3, 2),
            InvertedResidual(32, 3, 32, 3, 1),
            InvertedResidual(32, 3, 32, 3, 1),
            InvertedResidual(32, 3, 32, 3, 1),
            InvertedResidual(32, 6, 48, 3, 2),
            InvertedResidual(48, 3, 48, 3, 1),
            InvertedResidual(48, 3, 48, 3, 1),
            InvertedResidual(48, 3, 48, 5, 1),
            InvertedResidual(48, 6, 88, 3, 2),
            InvertedResidual(88, 3, 88, 3, 1),
            InvertedResidual(88, 6, 104, 5, 1),
            InvertedResidual(104, 3, 104, 3, 1),
            InvertedResidual(104, 3, 104, 3, 1),
            InvertedResidual(104, 3, 104, 3, 1),
            InvertedResidual(104, 6, 216, 5, 2),
            InvertedResidual(216, 3, 216, 5, 1),
            InvertedResidual(216, 3, 216, 5, 1),
            InvertedResidual(216, 3, 216, 3, 1),
            InvertedResidual(216, 6, 360, 5, 1),
        )
        self.last_block = nn.Sequential(
            nn.Conv2d(360, 1432, 1, 1, bias=False),
            nn.BatchNorm2d(1432),
            nn.ReLU6(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )
        self.output = nn.Linear(1432, num_classes)

    def forward(self, x):
        x = self.first_block(x)
        x = self.mb_blocks(x)
        x = self.last_block(x)
        x = self.output(x)
        return x


class ProxylessMobile(nn.Module):
    def __init__(self, num_classes=1000, small_input=False):
        super(ProxylessMobile, self).__init__()
        self.first_block = nn.Sequential(
            nn.Conv2d(3, 32, 3, 2 if not small_input else 1, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU6(inplace=True),
            nn.Conv2d(32, 32, 3, 1, 1, groups=40, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU6(inplace=True),
            nn.Conv2d(32, 16, 1, 1, bias=False),
            nn.BatchNorm2d(16)
        )
        self.mb_blocks = nn.Sequential(
            InvertedResidual(16, 3, 32, 5, 2),
            InvertedResidual(32, 3, 32, 3, 1),
            InvertedResidual(32, 3, 40, 7, 2),
            InvertedResidual(40, 3, 40, 3, 1),
            InvertedResidual(40, 3, 40, 5, 1),
            InvertedResidual(40, 3, 40, 5, 1),
            InvertedResidual(40, 6, 80, 7, 2),
            InvertedResidual(80, 3, 80, 5, 1),
            InvertedResidual(80, 3, 80, 5, 1),
            InvertedResidual(80, 3, 80, 5, 1),
            InvertedResidual(80, 6, 96, 5, 1),
            InvertedResidual(96, 3, 96, 5, 1),
            InvertedResidual(96, 3, 96, 5, 1),
            InvertedResidual(96, 3, 96, 5, 1),
            InvertedResidual(96, 6, 192, 7, 2),
            InvertedResidual(192, 6, 192, 7, 1),
            InvertedResidual(192, 3, 192, 7, 1),
            InvertedResidual(192, 3, 192, 7, 1),
            InvertedResidual(192, 6, 320, 7, 1),
        )
        self.last_block = nn.Sequential(
            nn.Conv2d(320, 1280, 1, 1, bias=False),
            nn.BatchNorm2d(1280),
            nn.ReLU6(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )
        self.output = nn.Linear(1280, num_classes)

    def forward(self, x):
        x = self.first_block(x)
        x = self.mb_blocks(x)
        x = self.last_block(x)
        x = self.output(x)
        return x
