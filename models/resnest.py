# -*- coding: utf-8 -*-
__all__ = ['resnest50']

import torch
from torch import nn


class SplitAttention2d(nn.Module):
    def __init__(self, channels, inter_channels, radix, groups):
        super(SplitAttention2d, self).__init__()
        self.radix = radix
        self.channels = channels
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, inter_channels, 1, groups=groups, bias=False)
        self.bn1 = nn.BatchNorm2d(inter_channels)
        self.fc2 = nn.Conv2d(inter_channels, channels * radix, 1, groups=groups)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        n, c, h, w = x.size()  # c = channels * radix
        sp = torch.reshape(x, (n, self.radix, self.channels, h, w))
        x = torch.sum(sp, dim=1)
        x = self.pool(x)
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.act(x)
        x = self.fc2(x).reshape((n, self.radix, self.channels, 1, 1))
        x = torch.softmax(x, dim=1)
        x = torch.sum(x * sp, dim=1)
        return x


class SplAtConv2d(nn.Module):
    def __init__(self, in_channels, channels, kernel_size, stride=(1, 1), padding=(0, 0),
                 dilation=(1, 1), groups=1, bias=True, radix=2, reduction_factor=4, **kwargs):
        super(SplAtConv2d, self).__init__()
        inter_channels = max(in_channels * radix // reduction_factor, 32)
        assert radix > 1
        self.radix = radix
        self.groups = groups
        self.channels = channels
        self.conv = nn.Conv2d(in_channels, channels * radix, kernel_size, stride, padding,
                              dilation, groups * radix, bias, **kwargs)
        self.bn = nn.BatchNorm2d(channels * radix)
        self.act = nn.ReLU(inplace=True)
        self.spaconv = SplitAttention2d(channels, inter_channels, radix, groups)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        x = self.spaconv(x)
        return x


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, radix=1, groups=1,
                 bottleneck_width=64, dilation=1):
        super(Bottleneck, self).__init__()
        group_width = int(planes * (bottleneck_width / 64.)) * groups
        self.radix = radix
        self.conv1 = nn.Conv2d(inplanes, group_width, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(group_width)
        self.conv2 = SplAtConv2d(group_width, group_width, kernel_size=3,
                                 stride=stride, padding=dilation, dilation=dilation,
                                 groups=groups, bias=False, radix=radix)
        self.conv3 = nn.Conv2d(group_width, planes * self.expansion, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.act = nn.ReLU(inplace=True)
        self.downsample = nn.Identity() if downsample is None else downsample

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act(out)

        out = self.conv2(out)

        out = self.conv3(out)
        out = self.bn3(out)

        identity = self.downsample(x)

        out += identity
        out = self.act(out)
        return out


class ResNet(nn.Module):
    def __init__(self, layers, radix=1, groups=1, bottleneck_width=64,
                 num_classes=1000):
        super(ResNet, self).__init__()
        self.groups = groups
        self.radix = radix
        self.bottleneck_width = bottleneck_width

        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.act = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(64, layers[0])
        self.layer2 = self._make_layer(128, layers[1], stride=2)
        self.layer3 = self._make_layer(256, layers[2], stride=2)
        self.layer4 = self._make_layer(512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(512 * Bottleneck.expansion, num_classes)

    def _make_layer(self, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * Bottleneck.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * Bottleneck.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * Bottleneck.expansion)
            )
        layers = []
        layers.append(Bottleneck(self.inplanes, planes, stride, downsample,
                                 self.radix, self.groups, self.bottleneck_width))
        self.inplanes = planes * Bottleneck.expansion
        for _ in range(1, blocks):
            layers.append(Bottleneck(self.inplanes, planes,
                                     radix=self.radix, groups=self.groups,
                                     bottleneck_width=self.bottleneck_width))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x


def resnest50(**kwargs):
    return ResNet([3, 4, 6, 3], radix=2, groups=1, bottleneck_width=64, **kwargs)
