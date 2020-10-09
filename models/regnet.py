# -*- coding: utf-8 -*-
__all__ = ['RegNetX200MF', 'RegNetX400MF', 'RegNetX600MF', 'RegNetX800MF',
           'RegNetX1_6GF', 'RegNetX3_2GF', 'RegNetX4_0GF', 'RegNetX6_4GF', 'RegNetX8_0GF',
           'RegNetY200MF', 'RegNetY400MF', 'RegNetY600MF', 'RegNetY800MF',
           'RegNetY1_6GF', 'RegNetY3_2GF', 'RegNetY4_0GF', 'RegNetY6_4GF', 'RegNetY8_0GF'
           ]

from torch import nn
from torchtoolbox.tools import make_divisible


class Stem(nn.Module):
    def __init__(self, in_c, out_c):
        super(Stem, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class SE(nn.Module):
    def __init__(self, in_c, reduction_ratio=0.25):
        super(SE, self).__init__()
        reducation_c = int(in_c * reduction_ratio)
        self.block = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_c, reducation_c, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(reducation_c, in_c, kernel_size=1, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        return x * self.block(x)


class Stage(nn.Module):
    def __init__(self, in_c, out_c, stride, bottleneck_ratio, group_width, reduction_ratio=0):
        super(Stage, self).__init__()
        width = make_divisible(out_c * bottleneck_ratio)
        groups = width // group_width

        self.block = nn.Sequential(
            nn.Conv2d(in_c, width, kernel_size=1, bias=False),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True),

            nn.Conv2d(width, width, kernel_size=3, stride=stride, padding=1, groups=groups, bias=False),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True),
            SE(width, reduction_ratio) if reduction_ratio != 0 else nn.Identity(),

            nn.Conv2d(width, out_c, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_c)
        )

        if in_c != out_c or stride != 1:
            self.skip_connection = nn.Sequential(
                nn.Conv2d(in_c, out_c, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_c)
            )
        else:
            self.skip_connection = nn.Identity()

        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        skip = self.skip_connection(x)
        x = self.block(x)
        x = self.act(x + skip)
        return x


class Head(nn.Module):
    def __init__(self, in_c, out_c):
        super(Head, self).__init__()
        self.block = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(in_c, out_c, bias=True)
        )

    def forward(self, x):
        return self.block(x)


class RegNet(nn.Module):
    def __init__(self, d, w, g, num_classes=1000, b=1, se=False):
        super(RegNet, self).__init__()
        self.reduction_ratio = 0.25 if se else 0
        self.bottleneck_ratio = b
        self.group_width = g
        stem_c = 32

        self.stem = Stem(3, stem_c)
        self.stage = nn.Sequential(
            self._make_layer(stem_c, w[0], d[0], 2),
            self._make_layer(w[0], w[1], d[1], 2),
            self._make_layer(w[1], w[2], d[2], 2),
            self._make_layer(w[2], w[3], d[3], 2))
        self.head = Head(w[3], num_classes)

    def _make_layer(self, in_c, out_c, blocks, stride=2):
        layers = []
        layers.append(Stage(in_c, out_c, stride, self.bottleneck_ratio,
                            self.group_width, self.reduction_ratio))
        for _ in range(1, blocks):
            layers.append(Stage(out_c, out_c, 1, self.bottleneck_ratio,
                                self.group_width, self.reduction_ratio))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.stem(x)
        x = self.stage(x)
        x = self.head(x)
        return x


_regnetx_config = {
    '200MF': {'d': [1, 1, 4, 7], 'w': [24, 56, 152, 368], 'g': 8},
    '400MF': {'d': [1, 2, 7, 12], 'w': [32, 64, 160, 384], 'g': 16},
    '600MF': {'d': [1, 3, 5, 7], 'w': [48, 96, 240, 528], 'g': 24},
    '800MF': {'d': [1, 3, 5, 7], 'w': [64, 128, 288, 672], 'g': 16},
    '1.6GF': {'d': [2, 4, 10, 2], 'w': [72, 168, 408, 912], 'g': 24},
    '3.2GF': {'d': [2, 6, 15, 2], 'w': [96, 192, 432, 1008], 'g': 48},
    '4.0GF': {'d': [2, 5, 14, 2], 'w': [80, 240, 560, 1360], 'g': 40},
    '6.4GF': {'d': [2, 4, 10, 1], 'w': [168, 392, 784, 1624], 'g': 56},
    '8.0GF': {'d': [2, 5, 15, 1], 'w': [80, 240, 720, 1920], 'g': 120},
    '12GF': {'d': [2, 5, 11, 1], 'w': [224, 448, 896, 2240], 'g': 112},
    '16GF': {'d': [2, 6, 13, 1], 'w': [256, 512, 896, 2048], 'g': 128},
    '32GF': {'d': [2, 7, 13, 1], 'w': [336, 672, 1344, 2520], 'g': 168},
}

_regnety_config = {
    '200MF': {'d': [1, 1, 4, 7], 'w': [24, 56, 152, 368], 'g': 8},
    '400MF': {'d': [1, 3, 6, 6], 'w': [48, 104, 208, 440], 'g': 8},
    '600MF': {'d': [1, 3, 7, 4], 'w': [48, 112, 256, 608], 'g': 16},
    '800MF': {'d': [1, 3, 8, 2], 'w': [64, 128, 320, 768], 'g': 16},
    '1.6GF': {'d': [2, 6, 17, 2], 'w': [48, 120, 336, 888], 'g': 24},
    '3.2GF': {'d': [2, 5, 13, 1], 'w': [72, 216, 576, 1512], 'g': 24},
    '4.0GF': {'d': [2, 6, 12, 2], 'w': [128, 192, 512, 1088], 'g': 64},
    '6.4GF': {'d': [2, 7, 14, 2], 'w': [144, 288, 576, 1296], 'g': 72},
    '8.0GF': {'d': [2, 4, 10, 1], 'w': [168, 448, 896, 2016], 'g': 56},
    '12GF': {'d': [2, 5, 11, 1], 'w': [224, 448, 896, 2240], 'g': 112},
    '16GF': {'d': [2, 4, 11, 1], 'w': [224, 448, 1232, 3024], 'g': 112},
    '32GF': {'d': [2, 5, 12, 1], 'w': [232, 696, 1392, 3712], 'g': 232},
}


def _regnet(name, b=1, se=False, **kwargs):
    config = _regnetx_config[name] if not se \
        else _regnety_config[name]

    d, w, g = config['d'], config['w'], config['g']
    return RegNet(d, w, g, b=b, se=se, **kwargs)


def RegNetX200MF(**kwargs):
    return _regnet('200MF', **kwargs)


def RegNetX400MF(**kwargs):
    return _regnet('400MF', **kwargs)


def RegNetX600MF(**kwargs):
    return _regnet('600MF', **kwargs)


def RegNetX800MF(**kwargs):
    return _regnet('800MF', **kwargs)


def RegNetX1_6GF(**kwargs):
    return _regnet('1.6GF', **kwargs)


def RegNetX3_2GF(**kwargs):
    return _regnet('3.2GF', **kwargs)


def RegNetX4_0GF(**kwargs):
    return _regnet('4.0GF', **kwargs)


def RegNetX6_4GF(**kwargs):
    return _regnet('6.4GF', **kwargs)


def RegNetX8_0GF(**kwargs):
    return _regnet('8.0GF', **kwargs)


def RegNetX12GF(**kwargs):
    return _regnet('12GF', **kwargs)


def RegNetX16GF(**kwargs):
    return _regnet('16GF', **kwargs)


def RegNetX32GF(**kwargs):
    return _regnet('32GF', **kwargs)


def RegNetY200MF(**kwargs):
    return _regnet('200MF', se=True, **kwargs)


def RegNetY400MF(**kwargs):
    return _regnet('400MF', se=True, **kwargs)


def RegNetY600MF(**kwargs):
    return _regnet('600MF', se=True, **kwargs)


def RegNetY800MF(**kwargs):
    return _regnet('800MF', se=True, **kwargs)


def RegNetY1_6GF(**kwargs):
    return _regnet('1.6GF', se=True, **kwargs)


def RegNetY3_2GF(**kwargs):
    return _regnet('3.2GF', se=True, **kwargs)


def RegNetY4_0GF(**kwargs):
    return _regnet('4.0GF', se=True, **kwargs)


def RegNetY6_4GF(**kwargs):
    return _regnet('6.4GF', se=True, **kwargs)


def RegNetY8_0GF(**kwargs):
    return _regnet('8.0GF', se=True, **kwargs)


def RegNetY12GF(**kwargs):
    return _regnet('12GF', se=True, **kwargs)


def RegNetY16GF(**kwargs):
    return _regnet('16GF', se=True, **kwargs)


def RegNetY32GF(**kwargs):
    return _regnet('32GF', se=True, **kwargs)
