# -*- coding: utf-8 -*-
# @Author  : DevinYang(pistonyang@gmail.com)

from torch import nn


class InvertedResidual(nn.Module):
    def __init__(self, in_c, expansion, out_c, kernel_size, stride):
        super(InvertedResidual, self).__init__()
        hidden_c = round(in_c * expansion)
        self.skip = stride == 1 and in_c == out_c
        self.act = nn.ReLU6(inplace=True)

        self.conv1 = nn.Conv2d(in_c, hidden_c, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(hidden_c)
        self.conv2 = nn.Conv2d(hidden_c, hidden_c, kernel_size, stride,
                               kernel_size // 2, groups=hidden_c, bias=False)
        self.bn2 = nn.BatchNorm2d(hidden_c)
        self.conv3 = nn.Conv2d(hidden_c, out_c, 1, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_c)

    def forward(self, x):
        skip = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.act(x)
        x = self.conv3(x)
        x = self.bn3(x)
        if self.skip:
            x = skip + x
        return x


class FairNasA(nn.Module):
    def __init__(self, num_classes=1000, small_input=False):
        super(FairNasC, self).__init__()
        self.first_block = nn.Sequential(
            nn.Conv2d(3, 32, 3, 2 if not small_input else 1, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU6(inplace=True),
            nn.Conv2d(32, 32, 3, 1, 1, groups=32, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU6(inplace=True),
            nn.Conv2d(32, 16, 1, 1, bias=False),
            nn.BatchNorm2d(16)
        )
        self.mb_blocks = nn.Sequential(
            InvertedResidual(16, 3, 32, 7, 2),
            InvertedResidual(32, 3, 32, 3, 1),
            InvertedResidual(32, 3, 40, 7, 2),
            InvertedResidual(40, 6, 40, 3, 1),
            InvertedResidual(40, 6, 40, 7, 1),
            InvertedResidual(40, 3, 40, 3, 1),
            InvertedResidual(40, 3, 80, 3, 2),
            InvertedResidual(80, 6, 80, 7, 1),
            InvertedResidual(80, 6, 80, 7, 1),
            InvertedResidual(80, 3, 80, 5, 1),
            InvertedResidual(80, 6, 96, 3, 1),
            InvertedResidual(96, 3, 96, 5, 1),
            InvertedResidual(96, 3, 96, 5, 1),
            InvertedResidual(96, 3, 96, 3, 1),
            InvertedResidual(96, 6, 192, 3, 2),
            InvertedResidual(192, 6, 192, 7, 1),
            InvertedResidual(192, 6, 192, 3, 1),
            InvertedResidual(192, 6, 192, 7, 1),
            InvertedResidual(192, 6, 320, 5, 1),
        )
        self.last_block = nn.Sequential(
            nn.Conv2d(320, 1280, 1, 1, bias=False),
            nn.BatchNorm2d(1280),
            nn.ReLU6(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.output = nn.Linear(1280, num_classes)

    def forward(self, x):
        x = self.first_block(x)
        x = self.mb_blocks(x)
        x = self.last_block(x)
        x = x.view(x.size(0), -1)
        x = self.output(x)
        return x


class FairNasB(nn.Module):
    def __init__(self, num_classes=1000, small_input=False):
        super(FairNasC, self).__init__()
        self.first_block = nn.Sequential(
            nn.Conv2d(3, 32, 3, 2 if not small_input else 1, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU6(inplace=True),
            nn.Conv2d(32, 32, 3, 1, 1, groups=32, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU6(inplace=True),
            nn.Conv2d(32, 16, 1, 1, bias=False),
            nn.BatchNorm2d(16)
        )
        self.mb_blocks = nn.Sequential(
            InvertedResidual(16, 3, 32, 5, 2),
            InvertedResidual(32, 3, 32, 3, 1),
            InvertedResidual(32, 3, 40, 5, 2),
            InvertedResidual(40, 3, 40, 3, 1),
            InvertedResidual(40, 6, 40, 3, 1),
            InvertedResidual(40, 3, 40, 5, 1),
            InvertedResidual(40, 3, 80, 7, 2),
            InvertedResidual(80, 3, 80, 3, 1),
            InvertedResidual(80, 6, 80, 3, 1),
            InvertedResidual(80, 3, 80, 5, 1),
            InvertedResidual(80, 3, 96, 3, 1),
            InvertedResidual(96, 6, 96, 3, 1),
            InvertedResidual(96, 3, 96, 7, 1),
            InvertedResidual(96, 3, 96, 3, 1),
            InvertedResidual(96, 6, 192, 7, 2),
            InvertedResidual(192, 6, 192, 5, 1),
            InvertedResidual(192, 6, 192, 7, 1),
            InvertedResidual(192, 6, 192, 3, 1),
            InvertedResidual(192, 6, 320, 5, 1),
        )
        self.last_block = nn.Sequential(
            nn.Conv2d(320, 1280, 1, 1, bias=False),
            nn.BatchNorm2d(1280),
            nn.ReLU6(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.output = nn.Linear(1280, num_classes)

    def forward(self, x):
        x = self.first_block(x)
        x = self.mb_blocks(x)
        x = self.last_block(x)
        x = x.view(x.size(0), -1)
        x = self.output(x)
        return x


class FairNasC(nn.Module):
    def __init__(self, num_classes=1000, small_input=False):
        super(FairNasC, self).__init__()
        self.first_block = nn.Sequential(
            nn.Conv2d(3, 32, 3, 2 if not small_input else 1, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU6(inplace=True),
            nn.Conv2d(32, 32, 3, 1, 1, groups=32, bias=False),
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
            InvertedResidual(40, 3, 40, 3, 1),
            InvertedResidual(40, 3, 40, 3, 1),
            InvertedResidual(40, 3, 80, 3, 2),
            InvertedResidual(80, 3, 80, 3, 1),
            InvertedResidual(80, 3, 80, 3, 1),
            InvertedResidual(80, 6, 80, 3, 1),
            InvertedResidual(80, 3, 96, 3, 1),
            InvertedResidual(96, 3, 96, 3, 1),
            InvertedResidual(96, 3, 96, 3, 1),
            InvertedResidual(96, 3, 96, 3, 1),
            InvertedResidual(96, 6, 192, 7, 2),
            InvertedResidual(192, 6, 192, 7, 1),
            InvertedResidual(192, 6, 192, 3, 1),
            InvertedResidual(192, 6, 192, 3, 1),
            InvertedResidual(192, 6, 320, 5, 1),
        )
        self.last_block = nn.Sequential(
            nn.Conv2d(320, 1280, 1, 1, bias=False),
            nn.BatchNorm2d(1280),
            nn.ReLU6(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.output = nn.Linear(1280, num_classes)

    def forward(self, x):
        x = self.first_block(x)
        x = self.mb_blocks(x)
        x = self.last_block(x)
        x = x.view(x.size(0), -1)
        x = self.output(x)
        return x
