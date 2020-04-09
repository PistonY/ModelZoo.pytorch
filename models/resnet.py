# -*- coding: utf-8 -*-
# @Author  : DevinYang(pistonyang@gmail.com)


import torch.nn as nn
from collections import namedtuple
from torchtoolbox.nn.activation import Activation

__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152', 'resnext50_32x4d', 'resnext101_32x8d',
           'ResNetV2', 'resnet18v2', 'resnet34v2', 'resnet50v2', 'resnet101v2',
           'resnet152v2', 'resnext50v2_32x4d', 'resnext101v2_32x8d']


def conv3x3(in_planes, out_planes, stride=1, groups=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, groups=groups, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, norm_layer=None, activation=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')

        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.act = Activation(activation, auto_optimize=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = nn.Identity() if downsample is None else downsample
        self.stride = stride

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act(out)

        out = self.conv2(out)
        out = self.bn2(out)

        identity = self.downsample(x)

        out += identity
        out = self.act(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, norm_layer=None, activation=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups

        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.act = Activation(activation, auto_optimize=True)
        self.downsample = nn.Identity() if downsample is None else downsample

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.act(out)

        out = self.conv3(out)
        out = self.bn3(out)

        identity = self.downsample(x)

        out += identity
        out = self.act(out)

        return out


class BasicBlockV2(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, norm_layer=None, activation=None):
        super(BasicBlockV2, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')

        self.bn1 = norm_layer(inplanes)
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn2 = norm_layer(planes)
        self.conv2 = conv3x3(planes, planes)
        self.downsample = downsample
        self.act = Activation(activation, auto_optimize=True)

    def forward(self, x):
        identity = x
        out = self.bn1(x)
        out = self.act(out)
        if self.downsample is not None:
            identity = self.downsample(out)
        out = self.conv1(out)

        out = self.bn2(out)
        out = self.act(out)
        out = self.conv2(out)

        out += identity

        return out


class BottleneckV2(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, norm_layer=None, activation=None):
        super(BottleneckV2, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups

        self.bn1 = norm_layer(inplanes)
        self.conv1 = conv1x1(inplanes, width)
        self.bn2 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups)
        self.bn3 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.act = Activation(activation, auto_optimize=True)

        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = self.bn1(x)
        out = self.act(out)
        if self.downsample is not None:
            identity = self.downsample(out)
        out = self.conv1(out)

        out = self.bn2(out)
        out = self.act(out)
        out = self.conv2(out)

        out = self.bn3(out)
        out = self.act(out)
        out = self.conv3(out)

        out += identity
        return out


class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1000, groups=1, width_per_group=64,
                 norm_layer=None, activation='relu', dropout_rate=None, small_input=False):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self._activation = activation

        self.inplanes = 64

        self.groups = groups
        self.base_width = width_per_group
        if small_input:
            self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1,
                                   bias=False)
        else:
            self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                                   bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.act = Activation(activation, auto_optimize=True)
        if small_input:
            self.maxpool = nn.Identity()
        else:
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(dropout_rate, inplace=True) if dropout_rate is not None else nn.Identity()
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, blocks, stride=1):
        norm_layer = self._norm_layer
        downsample = None

        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, norm_layer, self._activation))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, norm_layer=norm_layer,
                                activation=self._activation))

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
        x = self.dropout(x)
        x = self.fc(x)

        return x


class ResNetV2(nn.Module):
    def __init__(self, block, layers, num_classes=1000, groups=1, width_per_group=64,
                 norm_layer=None, activation='relu', dropout_rate=None, small_input=False):
        super(ResNetV2, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        self._norm_layer = norm_layer
        self._activation = activation

        self.inplanes = 64

        self.groups = groups
        self.base_width = width_per_group
        if small_input:
            self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1,
                                   bias=False)
        else:
            self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                                   bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.act = Activation(activation, auto_optimize=True)

        if small_input:
            self.maxpool = nn.Identity()
        else:
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.bn_last = norm_layer(512 * block.expansion)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(dropout_rate, inplace=True) if dropout_rate is not None else nn.Identity()
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, blocks, stride=1):
        norm_layer = self._norm_layer
        downsample = None

        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, norm_layer, self._activation))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, norm_layer=norm_layer,
                                activation=self._activation))

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
        x = self.bn_last(x)
        x = self.act(x)

        x = self.avgpool(x)
        x = self.flatten(x)
        x = self.dropout(x)
        x = self.fc(x)

        return x


def _resnet(block, layers, version=1, **kwargs):
    assert version in (1, 2)

    if version == 1:
        model = ResNet(block, layers, **kwargs)
    else:
        model = ResNetV2(block, layers, **kwargs)
    return model


def resnet18(**kwargs):
    """Constructs a ResNet-18 model.

    """
    return _resnet(BasicBlock, [2, 2, 2, 2], **kwargs)


def resnet18v2(**kwargs):
    """Constructs a ResNet-18v2 model.

    """
    return _resnet(BasicBlockV2, [2, 2, 2, 2], 2, **kwargs)


def resnet34(**kwargs):
    """Constructs a ResNet-34 model.

    """
    return _resnet(BasicBlock, [3, 4, 6, 3], **kwargs)


def resnet34v2(**kwargs):
    """Constructs a ResNet-34v2 model.

    """
    return _resnet(BasicBlockV2, [3, 4, 6, 3], 2, **kwargs)


def resnet50(**kwargs):
    """Constructs a ResNet-50 model.

    """
    return _resnet(Bottleneck, [3, 4, 6, 3], **kwargs)


def resnet50v2(**kwargs):
    """Constructs a ResNet-50 model.

    """
    return _resnet(BottleneckV2, [3, 4, 6, 3], 2, **kwargs)


def resnet101(**kwargs):
    """Constructs a ResNet-101 model.

    """
    return _resnet(Bottleneck, [3, 4, 23, 3], **kwargs)


def resnet101v2(**kwargs):
    """Constructs a ResNet-101v2 model.

    """
    return _resnet(Bottleneck, [3, 4, 23, 3], 2, **kwargs)


def resnet152(**kwargs):
    """Constructs a ResNet-152 model.

    """
    return _resnet(Bottleneck, [3, 8, 36, 3], **kwargs)


def resnet152v2(**kwargs):
    """Constructs a ResNet-152v2 model.

    """
    return _resnet(Bottleneck, [3, 8, 36, 3], 2, **kwargs)


def resnext50_32x4d(**kwargs):
    """Constructs a ResNeXt-50 32x4d model.

    """
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 4
    return _resnet(Bottleneck, [3, 4, 6, 3], **kwargs)


def resnext50v2_32x4d(**kwargs):
    """Constructs a ResNeXt-50v2 32x4d model.

    """
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 4
    return _resnet(Bottleneck, [3, 4, 6, 3], 2, **kwargs)


def resnext101_32x8d(**kwargs):
    """Constructs a ResNeXt-101 32x8d model.

    """
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 8
    return _resnet(Bottleneck, [3, 4, 23, 3], **kwargs)


def resnext101v2_32x8d(**kwargs):
    """Constructs a ResNeXt-101v2 32x8d model.

    """
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 8
    return _resnet(Bottleneck, [3, 4, 23, 3], 2, **kwargs)
