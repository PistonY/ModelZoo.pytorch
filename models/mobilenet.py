# -*- coding: utf-8 -*-
# @Author  : DevinYang(pistonyang@gmail.com)
__all__ = ['MobileNetV1', 'MobileNetV2', 'MobileNetV3_Large', 'MobileNetV3_Small']

from functools import partial
from torch import nn
import math
import torch.nn.functional as F


def make_divisible(x, divisible_by=8):
    return int(math.ceil(x * 1. / divisible_by) * divisible_by)


class HardSwish(nn.Module):
    def __init__(self, inplace=False):
        super(HardSwish, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return x * F.relu6(x + 3., inplace=self.inplace) / 6.


class HardSigmoid(nn.Module):
    def __init__(self, inplace=False):
        super(HardSigmoid, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return F.relu6(x + 3., inplace=self.inplace) / 6.


class SE_Module(nn.Module):
    def __init__(self, channels, reduction=4):
        super(SE_Module, self).__init__()
        reduction_c = make_divisible(channels // reduction)
        self.out = nn.Sequential(
            nn.Conv2d(channels, reduction_c, 1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduction_c, channels, 1, bias=True),
            HardSigmoid()
        )

    def forward(self, x):
        y = F.adaptive_avg_pool2d(x, 1)
        y = self.out(y)
        return x * y


class Activation(nn.Module):
    def __init__(self, act_type, inplace=False):
        super(Activation, self).__init__()
        if act_type == 'relu':
            self.act = nn.ReLU(inplace=inplace)
        elif act_type == 'relu6':
            self.act = nn.ReLU6(inplace=inplace)
        elif act_type == 'h_swish':
            self.act = HardSwish(inplace=inplace)
        else:
            raise NotImplementedError('{} activation is not implemented.'.format(act_type))

    def forward(self, x):
        return self.act(x)


class MobileNetBottleneck(nn.Module):
    def __init__(self, in_c, expansion, out_c, kernel_size, stride, se=False,
                 activation='relu6', first_conv=True, skip=True, linear=True):
        super(MobileNetBottleneck, self).__init__()

        self.act = Activation(activation, inplace=True)
        hidden_c = round(in_c * expansion)
        self.linear = linear
        self.skip = stride == 1 and in_c == out_c and skip

        seq = []
        if first_conv and in_c != hidden_c:
            seq.append(nn.Conv2d(in_c, hidden_c, 1, 1, bias=False))
            seq.append(nn.BatchNorm2d(hidden_c))
            seq.append(Activation(activation, inplace=True))
        seq.append(nn.Conv2d(hidden_c, hidden_c, kernel_size, stride,
                             kernel_size // 2, groups=hidden_c, bias=False))
        seq.append(nn.BatchNorm2d(hidden_c))
        seq.append(Activation(activation, inplace=True))
        if se:
            seq.append(SE_Module(hidden_c))
        seq.append(nn.Conv2d(hidden_c, out_c, 1, 1, bias=False))
        seq.append(nn.BatchNorm2d(out_c))

        self.seq = nn.Sequential(*seq)

    def forward(self, x):
        skip = x
        x = self.seq(x)
        if self.skip:
            x = skip + x
        if not self.linear:
            x = self.act(x)
        return x


class MobileNetV1(nn.Module):
    def __init__(self, num_classes=1000, small_input=False):
        super(MobileNetV1, self).__init__()
        self.first_block = nn.Sequential(
            nn.Conv2d(3, 32, 3, 2 if not small_input else 1, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        MB1_Bottleneck = partial(MobileNetBottleneck, first_conv=False,
                                 activation='relu', skip=False, linear=False)
        self.mb_block = nn.Sequential(
            MB1_Bottleneck(32, 1, 64, 3, 1),
            MB1_Bottleneck(64, 1, 128, 3, 2),
            MB1_Bottleneck(128, 1, 128, 3, 1),
            MB1_Bottleneck(128, 1, 256, 3, 2),
            MB1_Bottleneck(256, 1, 256, 3, 1),
            MB1_Bottleneck(256, 1, 512, 3, 2),
            MB1_Bottleneck(512, 1, 512, 3, 1),
            MB1_Bottleneck(512, 1, 512, 3, 1),
            MB1_Bottleneck(512, 1, 512, 3, 1),
            MB1_Bottleneck(512, 1, 512, 3, 1),
            MB1_Bottleneck(512, 1, 512, 3, 1),
            MB1_Bottleneck(512, 1, 1024, 3, 2),
            MB1_Bottleneck(1024, 1, 1024, 3, 1),
        )
        self.last_block = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )
        self.output = nn.Linear(1024, num_classes)

    def forward(self, x):
        x = self.first_block(x)
        x = self.mb_block(x)
        x = self.last_block(x)
        x = self.output(x)
        return x


class MobileNetV2(nn.Module):
    def __init__(self, num_classes=1000, small_input=False):
        super(MobileNetV2, self).__init__()
        self.first_block = nn.Sequential(
            nn.Conv2d(3, 32, 3, 2 if not small_input else 1, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU6(inplace=True),
            nn.Conv2d(32, 32, 3, 1, 1, groups=32, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU6(inplace=True),
            nn.Conv2d(32, 16, 1, 1, bias=False),
            nn.BatchNorm2d(16),
        )
        self.mb_block = nn.Sequential(
            MobileNetBottleneck(16, 6, 24, 3, 2),
            MobileNetBottleneck(24, 6, 24, 3, 1),
            MobileNetBottleneck(24, 6, 32, 3, 2),
            MobileNetBottleneck(32, 6, 32, 3, 1),
            MobileNetBottleneck(32, 6, 32, 3, 1),
            MobileNetBottleneck(32, 6, 64, 3, 2),
            MobileNetBottleneck(64, 6, 64, 3, 1),
            MobileNetBottleneck(64, 6, 64, 3, 1),
            MobileNetBottleneck(64, 6, 64, 3, 1),
            MobileNetBottleneck(64, 6, 96, 3, 1),
            MobileNetBottleneck(96, 6, 96, 3, 1),
            MobileNetBottleneck(96, 6, 96, 3, 1),
            MobileNetBottleneck(96, 6, 160, 3, 2),
            MobileNetBottleneck(160, 6, 160, 3, 1),
            MobileNetBottleneck(160, 6, 160, 3, 1),
            MobileNetBottleneck(160, 6, 320, 3, 1),
        )
        self.last_block = nn.Sequential(
            nn.Conv2d(320, 1280, 1, 1, bias=False),
            nn.BatchNorm2d(1280),
            nn.ReLU6(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten()
        )
        self.output = nn.Linear(1280, num_classes)

    def forward(self, x):
        x = self.first_block(x)
        x = self.mb_block(x)
        x = self.last_block(x)
        x = self.output(x)
        return x


class MobileNetV3_Large(nn.Module):
    def __init__(self, num_classes=1000, small_input=False):
        super(MobileNetV3_Large, self).__init__()
        self.first_block = nn.Sequential(
            nn.Conv2d(3, 16, 3, 2 if not small_input else 1, 1, bias=False),
            nn.BatchNorm2d(16),
            HardSwish(inplace=True),
        )
        self.mb_block = nn.Sequential(
            MobileNetBottleneck(16, 1, 16, 3, 1, False, 'relu'),
            MobileNetBottleneck(16, 4, 24, 3, 2, False, 'relu'),
            MobileNetBottleneck(24, 3, 24, 3, 1, False, 'relu'),
            MobileNetBottleneck(24, 3, 40, 5, 2, True, 'relu'),
            MobileNetBottleneck(40, 3, 40, 5, 1, True, 'relu'),
            MobileNetBottleneck(40, 3, 40, 5, 1, True, 'relu'),
            MobileNetBottleneck(40, 6, 80, 3, 2, False, 'h_swish'),
            MobileNetBottleneck(80, 2.5, 80, 3, 1, False, 'h_swish'),
            MobileNetBottleneck(80, 2.3, 80, 3, 1, False, 'h_swish'),
            MobileNetBottleneck(80, 2.3, 80, 3, 1, False, 'h_swish'),
            MobileNetBottleneck(80, 6, 112, 3, 1, True, 'h_swish'),
            MobileNetBottleneck(112, 6, 112, 3, 1, True, 'h_swish'),
            MobileNetBottleneck(112, 6, 160, 5, 2, True, 'h_swish'),
            MobileNetBottleneck(160, 6, 160, 5, 1, True, 'h_swish'),
            MobileNetBottleneck(160, 6, 160, 5, 1, True, 'h_swish'),
        )
        self.last_block = nn.Sequential(
            nn.Conv2d(160, 960, 1, bias=False),
            nn.BatchNorm2d(960),
            HardSwish(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(960, 1280, 1),
            HardSwish(),
            nn.Flatten(),
        )
        self.output = nn.Linear(1280, num_classes)

    def forward(self, x):
        x = self.first_block(x)
        x = self.mb_block(x)
        x = self.last_block(x)
        x = self.output(x)
        return x


class MobileNetV3_Small(nn.Module):
    def __init__(self, num_classes=1000, small_input=False):
        super(MobileNetV3_Small, self).__init__()
        self.first_block = nn.Sequential(
            nn.Conv2d(3, 16, 3, 2 if not small_input else 1, 1, bias=False),
            nn.BatchNorm2d(16),
            HardSwish(inplace=True),
        )
        self.mb_block = nn.Sequential(
            MobileNetBottleneck(16, 1, 16, 3, 2, True, 'relu'),
            MobileNetBottleneck(16, 4.5, 24, 3, 2, False, 'relu'),
            MobileNetBottleneck(24, 88 / 24, 24, 3, 1, False, 'relu'),
            MobileNetBottleneck(24, 4, 40, 5, 2, True, 'h_swish'),
            MobileNetBottleneck(40, 6, 40, 5, 1, True, 'h_swish'),
            MobileNetBottleneck(40, 6, 40, 5, 1, True, 'h_swish'),
            MobileNetBottleneck(40, 3, 48, 5, 1, True, 'h_swish'),
            MobileNetBottleneck(48, 3, 48, 5, 1, True, 'h_swish'),
            MobileNetBottleneck(48, 6, 96, 5, 2, True, 'h_swish'),
            MobileNetBottleneck(96, 6, 96, 5, 1, True, 'h_swish'),
            MobileNetBottleneck(96, 6, 96, 5, 1, True, 'h_swish'),
        )
        self.last_block = nn.Sequential(
            nn.Conv2d(96, 576, 1, bias=False),
            nn.BatchNorm2d(576),
            HardSwish(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(576, 1280, 1),
            HardSwish(),
            nn.Flatten(),
        )
        self.output = nn.Linear(1280, num_classes)

    def forward(self, x):
        x = self.first_block(x)
        x = self.mb_block(x)
        x = self.last_block(x)
        x = self.output(x)
        return x
