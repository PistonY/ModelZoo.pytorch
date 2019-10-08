# -*- coding: utf-8 -*-
# @Author  : DevinYang(pistonyang@gmail.com)


__all__ = ['EfficientNet', 'EfficientNet_B0', 'EfficientNet_B1', 'EfficientNet_B2',
           'EfficientNet_B3', 'EfficientNet_B4', 'EfficientNet_B5', 'EfficientNet_B6',
           'EfficientNet_B7']

import math
import torch
from torch import nn


class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class DropConnect(nn.Module):
    def __init__(self, ratio):
        super().__init__()
        self.ratio = 1.0 - ratio

    def forward(self, x):
        if not self.training:
            return x

        random_tensor = self.ratio
        random_tensor += torch.rand([x.shape[0], 1, 1, 1], dtype=torch.float, device=x.device)
        random_tensor.requires_grad_(False)
        return x / self.ratio * random_tensor.floor()


def _conv_bn(in_c, out_c, kernel_size, stride=1, groups=1,
             eps=1e-5, momentum=0.1, use_act=False):
    layer = []
    layer.append(nn.Conv2d(in_c, out_c, kernel_size, stride, kernel_size // 2, groups=groups, bias=False))
    layer.append(nn.BatchNorm2d(out_c, eps, momentum))
    if use_act:
        layer.append(Swish())
    return nn.Sequential(*layer)


class SEModule(nn.Module):
    def __init__(self, in_, squeeze_ch):
        super().__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_, squeeze_ch, kernel_size=1, stride=1, padding=0, bias=True),
            Swish(),
            nn.Conv2d(squeeze_ch, in_, kernel_size=1, stride=1, padding=0, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.se(x)


class MBConv(nn.Module):
    def __init__(self, in_c, out_c, expand,
                 kernel_size, stride, se_ratio,
                 dc_ratio):
        super().__init__()
        exp_c = in_c * expand
        self.layer1 = _conv_bn(in_c, exp_c, 1, use_act=True) if expand != 1 else nn.Identity()
        self.layer2 = _conv_bn(exp_c, exp_c, kernel_size, stride,
                               groups=exp_c, use_act=True)
        self.se_layer = SEModule(exp_c, int(in_c * se_ratio)) if se_ratio > 0 else nn.Identity()
        self.layer3 = _conv_bn(exp_c, out_c, 1)
        self.skip = True if stride == 1 and in_c == out_c else False
        self.dropconnect = DropConnect(dc_ratio) if self.skip and dc_ratio > 0 else nn.Identity()

    def forward(self, inputs):
        x = self.layer1(inputs)
        x = self.layer2(x)
        x = self.se_layer(x)
        x = self.layer3(x)
        if self.skip:
            x = self.dropconnect(x) + inputs
        return x


class EfficientNet(nn.Module):
    def __init__(self, width_coeff, depth_coeff,
                 depth_div=8, min_depth=None,
                 dropout_rate=0., drop_connect_rate=0,
                 num_classes=1000, small_input=False):
        super().__init__()
        min_depth = min_depth or depth_div
        self.first_conv = _conv_bn(3, 32, 3, 2 if not small_input else 1, use_act=True)

        def renew_ch(x):
            if not width_coeff:
                return x

            # new_x = x * width_coeff
            new_x = max(min_depth, int(x + depth_div / 2) // depth_div * depth_div)
            if new_x < 0.9 * new_x:
                new_x += depth_div
            return new_x

        def renew_repeat(x):
            return int(math.ceil(x * depth_coeff))

        self.blocks = nn.Sequential(
            self._make_layer(renew_ch(32), renew_ch(16), 1, 3, 1, renew_repeat(1), 0.25, drop_connect_rate),
            self._make_layer(renew_ch(16), renew_ch(24), 6, 3, 2, renew_repeat(2), 0.25, drop_connect_rate),
            self._make_layer(renew_ch(24), renew_ch(40), 6, 5, 2, renew_repeat(2), 0.25, drop_connect_rate),
            self._make_layer(renew_ch(40), renew_ch(80), 6, 3, 2, renew_repeat(3), 0.25, drop_connect_rate),
            self._make_layer(renew_ch(80), renew_ch(112), 6, 5, 1, renew_repeat(3), 0.25, drop_connect_rate),
            self._make_layer(renew_ch(112), renew_ch(192), 6, 5, 2, renew_repeat(4), 0.25, drop_connect_rate),
            self._make_layer(renew_ch(192), renew_ch(320), 6, 3, 1, renew_repeat(1), 0.25, drop_connect_rate),
        )
        self.last_process = nn.Sequential(
            *_conv_bn(renew_ch(320), renew_ch(1280), 1, use_act=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Dropout2d(dropout_rate, True) if dropout_rate > 0 else nn.Identity(),
        )
        self.output = nn.Linear(renew_ch(1280), num_classes)

    def _make_layer(self, in_c, out_c, expand, kernel_size, stride, repeats, se_ratio, drop_connect_ratio):
        layers = []
        layers.append(MBConv(in_c, out_c, expand, kernel_size, stride, se_ratio, drop_connect_ratio))
        for _ in range(repeats - 1):
            layers.append(MBConv(out_c, out_c, expand, kernel_size, 1, se_ratio, drop_connect_ratio))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.first_conv(x)
        x = self.blocks(x)
        x = self.last_process(x)
        x = x.view(x.shape[0], -1)
        x = self.output(x)
        return x


def EfficientNet_B0(num_classes=1000, **kwargs):
    model = EfficientNet(1., 1., dropout_rate=0, num_classes=num_classes, **kwargs)
    return model


def EfficientNet_B1(num_classes=1000, **kwargs):
    # input size should be 240(~1.07x)
    model = EfficientNet(1., 1.1, dropout_rate=0.2, num_classes=num_classes, **kwargs)
    return model


def EfficientNet_B2(num_classes=1000, **kwargs):
    # input size should be 260(~1.16x)
    model = EfficientNet(1.1, 1.2, dropout_rate=0.3, num_classes=num_classes, **kwargs)
    return model


def EfficientNet_B3(num_classes=1000, **kwargs):
    # input size should be 300(~1.34x)
    model = EfficientNet(1.2, 1.4, dropout_rate=0.3, num_classes=num_classes, **kwargs)
    return model


def EfficientNet_B4(num_classes=1000, **kwargs):
    # input size should be 380(~1.70x)
    model = EfficientNet(1.4, 1.8, dropout_rate=0.4, num_classes=num_classes, **kwargs)
    return model


def EfficientNet_B5(num_classes=1000, **kwargs):
    # input size should be 456(~2.036x)
    model = EfficientNet(1.6, 2.2, dropout_rate=0.4, num_classes=num_classes, **kwargs)
    return model


def EfficientNet_B6(num_classes=1000, **kwargs):
    # input size should be 528(~2.357x)
    model = EfficientNet(1.8, 2.6, dropout_rate=0.5, num_classes=num_classes, **kwargs)
    return model


def EfficientNet_B7(num_classes=1000, **kwargs):
    # input size should be 600(~2.679x)
    model = EfficientNet(2., 3.1, dropout_rate=0.5, num_classes=num_classes, **kwargs)
    return model
