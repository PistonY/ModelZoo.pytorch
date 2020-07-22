# -*- coding: utf-8 -*-
# @Author  : DevinYang(pistonyang@gmail.com)
"""This file is totally same as oct_resnet, I just want to avoid if/else in forward."""

__all__ = ['OctResnet', 'oct_resnet50', 'oct_resnet101', 'oct_resnet152',
           'oct_resnet50_32x4d', 'oct_resnet101_32x8d']

from torch import nn
from torchtoolbox.nn import AdaptiveSequential
from module import OctConv, OctConvFirst, OctConvLast


class fs_bn(nn.Module):
    def __init__(self, channels, alpha):
        super().__init__()
        h_out = int((1 - alpha) * channels)
        l_out = int(alpha * channels)

        self.h_bn = nn.BatchNorm2d(h_out)
        self.l_bn = nn.BatchNorm2d(l_out)

    def forward(self, x_h, x_l=None):
        y_h = self.h_bn(x_h)
        y_l = self.l_bn(x_l)
        return y_h, y_l


class fs_relu(nn.Module):
    def __init__(self):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x_h, x_l=None):
        y_h = self.relu(x_h)
        y_l = self.relu(x_l)
        return y_h, y_l


class OctBottleneck_First(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, alpha, stride=1, groups=1, base_width=64):
        super(OctBottleneck_First, self).__init__()
        width = int(planes * (base_width / 64.)) * groups

        self.conv1 = OctConvFirst(inplanes, width, alpha, 1, bias=False)
        self.bn1 = fs_bn(width, alpha)
        self.conv2 = OctConv(width, width, alpha, 3, 1, 1, groups, False)
        self.bn2 = fs_bn(width, alpha)
        self.conv3 = OctConv(width, planes * self.expansion, alpha, 1, bias=False)
        self.bn3 = fs_bn(planes * self.expansion, alpha)

        self.relu = fs_relu()
        if stride != 1 or inplanes != planes * self.expansion:
            self.downsample = AdaptiveSequential(
                OctConvFirst(inplanes, planes * self.expansion, alpha,
                             1, stride=stride, bias=False),
                fs_bn(planes * self.expansion, alpha)
            )
        else:
            self.downsample = nn.Identity()

    def forward(self, x):
        r_h, r_l = self.downsample(x)
        x_h, x_l = self.conv1(x)
        x_h, x_l = self.bn1(x_h, x_l)
        x_h, x_l = self.relu(x_h, x_l)

        x_h, x_l = self.conv2(x_h, x_l)
        x_h, x_l = self.bn2(x_h, x_l)
        x_h, x_l = self.relu(x_h, x_l)

        x_h, x_l = self.conv3(x_h, x_l)
        x_h, x_l = self.bn3(x_h, x_l)

        y_h, y_l = self.relu(x_h + r_h, x_l + r_l)

        return y_h, y_l


class OctBottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, alpha, stride=1, groups=1, base_width=64):
        super(OctBottleneck, self).__init__()
        width = int(planes * (base_width / 64.)) * groups

        self.conv1 = OctConv(inplanes, width, alpha, 1, bias=False)
        self.bn1 = fs_bn(width, alpha)
        self.conv2 = OctConv(width, width, alpha, 3, 1, stride, groups, False)
        self.bn2 = fs_bn(width, alpha)
        self.conv3 = OctConv(width, planes * self.expansion, alpha, 1, bias=False)
        self.bn3 = fs_bn(planes * self.expansion, alpha)

        self.relu = fs_relu()
        if stride != 1 or inplanes != planes * self.expansion:
            self.downsample = AdaptiveSequential(
                OctConv(inplanes, planes * self.expansion, alpha,
                        1, stride=stride, bias=False),
                fs_bn(planes * self.expansion, alpha)
            )
        else:
            self.downsample = AdaptiveSequential()

    def forward(self, x_h, x_l):
        r_h, r_l = self.downsample(x_h, x_l)
        x_h, x_l = self.conv1(x_h, x_l)
        x_h, x_l = self.bn1(x_h, x_l)
        x_h, x_l = self.relu(x_h, x_l)

        x_h, x_l = self.conv2(x_h, x_l)
        x_h, x_l = self.bn2(x_h, x_l)
        x_h, x_l = self.relu(x_h, x_l)

        x_h, x_l = self.conv3(x_h, x_l)
        x_h, x_l = self.bn3(x_h, x_l)

        y_h, y_l = self.relu(x_h + r_h, x_l + r_l)
        return y_h, y_l


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, groups=1, base_width=64):
        super(Bottleneck, self).__init__()
        width = int(planes * (base_width / 64.)) * groups

        self.conv1 = nn.Conv2d(inplanes, width, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(width)
        self.conv2 = nn.Conv2d(width, width, 3, stride, 1, groups=groups, bias=False)
        self.bn2 = nn.BatchNorm2d(width)
        self.conv3 = nn.Conv2d(width, planes * self.expansion, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)

        self.relu = nn.ReLU()
        if stride != 1 or inplanes != planes * self.expansion:
            self.downsample = AdaptiveSequential(
                nn.Conv2d(inplanes, planes * self.expansion,
                          1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * self.expansion)
            )
        else:
            self.downsample = nn.Identity()

    def forward(self, x):
        r = self.downsample(x)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.conv3(x)
        x = self.bn3(x)

        y = self.relu(x + r)
        return y


class OctBottleneck_Last(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, alpha, stride=1, groups=1, base_width=64):
        super(OctBottleneck_Last, self).__init__()
        width = int(planes * (base_width / 64.)) * groups

        self.conv1 = OctConvLast(inplanes, width, alpha, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(width)
        self.conv2 = nn.Conv2d(width, width, 3, stride, 1, groups=groups, bias=False)
        self.bn2 = nn.BatchNorm2d(width)
        self.conv3 = nn.Conv2d(width, planes * self.expansion, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion, alpha)

        self.relu = nn.ReLU()
        if stride != 1 or inplanes != planes * self.expansion:
            self.downsample = AdaptiveSequential(
                OctConvLast(inplanes, planes * self.expansion, alpha,
                            1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * self.expansion)
            )
        else:
            self.downsample = nn.Identity()

    def forward(self, x_h, x_l):
        r = self.downsample(x_h, x_l)
        x = self.conv1(x_h, x_l)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.conv3(x)
        x = self.bn3(x)

        x = self.relu(x + r)
        return x


class OctResnet(nn.Module):
    def __init__(self, alpha, layers, num_classes=1000, groups=1, width_per_group=64):
        super(OctResnet, self).__init__()
        self.inplanes = 64
        self.groups = groups
        self.alpha = alpha
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(64, layers[0], 1, 'start')
        self.layer2 = self._make_layer(128, layers[1], 2)
        self.layer3 = self._make_layer(256, layers[2], 2)
        self.layer4 = self._make_layer(512, layers[3], 2, 'end')

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * OctBottleneck.expansion, num_classes)
        )

    def _make_layer(self, planes, blocks, stride=1, status='normal'):
        assert status in ('start', 'normal', 'end')
        layers = []
        if status == 'start':
            layers.append(OctBottleneck_First(self.inplanes, planes, self.alpha, stride,
                                              self.groups, self.base_width))
        elif status == 'normal':
            layers.append(OctBottleneck(self.inplanes, planes, self.alpha, stride,
                                        self.groups, self.base_width))
        else:
            layers.append(OctBottleneck_Last(self.inplanes, planes, self.alpha, stride,
                                             self.groups, self.base_width))
        self.inplanes = planes * OctBottleneck.expansion
        for _ in range(1, blocks):
            if status != 'end':
                layers.append(OctBottleneck(self.inplanes, planes, self.alpha, 1,
                                            self.groups, self.base_width))
            else:
                layers.append(Bottleneck(self.inplanes, planes, 1,
                                         self.groups, self.base_width))
            return AdaptiveSequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x_h, x_l = self.layer1(x)
        x_h, x_l = self.layer2(x_h, x_l)
        x_h, x_l = self.layer3(x_h, x_l)
        x = self.layer4(x_h, x_l)

        x = self.avgpool(x)
        x = self.fc(x)
        return x


def oct_resnet50(alpha, **kwargs):
    return OctResnet(alpha, [3, 4, 6, 3], **kwargs)


def oct_resnet101(alpha, **kwargs):
    return OctResnet(alpha, [3, 4, 23, 3], **kwargs)


def oct_resnet152(alpha, **kwargs):
    return OctResnet(alpha, [3, 8, 36, 3], **kwargs)


def oct_resnet50_32x4d(alpha, **kwargs):
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 4
    return OctResnet(alpha, [3, 4, 6, 3], **kwargs)


def oct_resnet101_32x8d(alpha, **kwargs):
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 8
    return OctResnet(alpha, [3, 4, 32, 3], **kwargs)
