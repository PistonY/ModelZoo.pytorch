# -*- coding: utf-8 -*-
# @Author  : DevinYang(pistonyang@gmail.com)

import torch
from torch import nn

__all__ = ['EvoResNet', 'evo_resnet18', 'evo_resnet34', 'evo_resnet50', 'evo_resnet101',
           'evo_resnet152', 'evo_resnext101_32x8d', 'evo_resnext50_32x4d']


def instance_std(x, eps=1e-5):
    var = torch.var(x, dim=(2, 3), keepdim=True)
    std = torch.sqrt(var + eps)
    return std


def group_std(x: torch.Tensor, groups=32, eps=1e-5):
    n, c, h, w = x.size()
    x = torch.reshape(x, (n, groups, c // groups, h, w))
    var = torch.var(x, dim=(2, 3, 4), keepdim=True)
    std = torch.sqrt(var + eps)
    return torch.reshape(std, (n, c, h, w))


def evo_norm(x, prefix, running_var, v, weight, bias,
             training, momentum, eps=0.1, groups=32):
    if prefix == 'b0':
        if training:
            var = torch.var(x, dim=(0, 2, 3), keepdim=True)
            running_var.mul_(momentum)
            running_var.add_((1 - momentum) * var)
        else:
            var = running_var
        if v is not None:
            # print(var.shape, x.shape, v.shape)
            # _ = v * x
            den = torch.max((var + eps).sqrt(), v * x + instance_std(x, eps))
            x = x / den * weight + bias
        else:
            x = x * weight + bias
    elif prefix == 's0':
        if v is not None:
            x = x * torch.sigmoid(v * x) / group_std(x, groups, eps) * weight + bias
        else:
            x = x * weight + bias
    else:
        raise NotImplementedError
    return x


class _EvoNorm(nn.Module):
    def __init__(self, prefix, num_features, eps=1e-5, momentum=0.9, groups=32,
                 affine=True):
        super(_EvoNorm, self).__init__()
        assert prefix in ('s0', 'b0')
        self.prefix = prefix
        self.groups = groups
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        if self.affine:
            self.weight = nn.Parameter(torch.Tensor(1, num_features, 1, 1))
            self.bias = nn.Parameter(torch.Tensor(1, num_features, 1, 1))
            self.v = nn.Parameter(torch.Tensor(1, num_features, 1, 1))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
            self.register_parameter('v', None)
        self.register_buffer('running_var', torch.ones(1, num_features, 1, 1))
        self.reset_parameters()

    def reset_parameters(self):
        if self.affine:
            torch.nn.init.ones_(self.weight)
            torch.nn.init.zeros_(self.bias)
            torch.nn.init.ones_(self.v)

    def _check_input_dim(self, x):
        if x.dim() != 4:
            raise ValueError('expected 4D input (got {}D input)'
                             .format(x.dim()))

    def forward(self, x):
        self._check_input_dim(x)
        return evo_norm(x, self.prefix, self.running_var, self.v,
                        self.weight, self.bias, self.training,
                        self.momentum, self.eps, self.groups)


class EvoNormB0(_EvoNorm):
    def __init__(self, num_features, eps=1e-5, momentum=0.9, affine=True):
        super(EvoNormB0, self).__init__('b0', num_features, eps, momentum,
                                        affine=affine)


class EvoNormS0(_EvoNorm):
    def __init__(self, num_features, groups=32, affine=True):
        super(EvoNormS0, self).__init__('s0', num_features, groups=groups,
                                        affine=affine)


def conv3x3(in_planes, out_planes, stride=1, groups=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, groups=groups, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64):
        super(Bottleneck, self).__init__()
        width = int(planes * (base_width / 64.)) * groups
        self.evonorm1 = EvoNormB0(inplanes)
        self.conv1 = conv1x1(inplanes, width)
        self.evonorm2 = EvoNormB0(width)
        self.conv2 = conv3x3(width, width, stride, groups)
        self.evonorm3 = EvoNormB0(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = self.evonorm1(x)
        if self.downsample is not None:
            identity = self.downsample(out)
        out = self.conv1(out)
        out = self.evonorm2(out)
        out = self.conv2(out)
        out = self.evonorm3(out)
        out = self.conv3(out)
        out += identity
        return out


class EvoResNet(nn.Module):
    def __init__(self, layers, num_classes=1000, groups=1, width_per_group=64,
                 dropout_rate=None, small_input=False):
        super(EvoResNet, self).__init__()
        self.inplanes = 64

        self.groups = groups
        self.base_width = width_per_group
        if small_input:
            self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1,
                                   bias=False)
        else:
            self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                                   bias=False)
        self.evonorm1 = EvoNormB0(self.inplanes)
        if small_input:
            self.maxpool = nn.Identity()
        else:
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(64, layers[0])
        self.layer2 = self._make_layer(128, layers[1], stride=2)
        self.layer3 = self._make_layer(256, layers[2], stride=2)
        self.layer4 = self._make_layer(512, layers[3], stride=2)

        self.evonorm2 = EvoNormB0(512 * Bottleneck.expansion)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(dropout_rate, inplace=True) if dropout_rate is not None else nn.Identity()
        self.fc = nn.Linear(512 * Bottleneck.expansion, num_classes)

    def _make_layer(self, planes, blocks, stride=1):
        downsample = None

        if stride != 1 or self.inplanes != planes * Bottleneck.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * Bottleneck.expansion, stride),
            )

        layers = []
        layers.append(Bottleneck(self.inplanes, planes, stride, downsample, self.groups,
                                 self.base_width))
        self.inplanes = planes * Bottleneck.expansion
        for _ in range(1, blocks):
            layers.append(Bottleneck(self.inplanes, planes, groups=self.groups,
                                     base_width=self.base_width))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.evonorm1(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.evonorm2(x)

        x = self.avgpool(x)
        x = self.flatten(x)
        x = self.dropout(x)
        x = self.fc(x)

        return x


def evo_resnet18(**kwargs):
    """Constructs a ResNet-18 model.

    """
    return EvoResNet([2, 2, 2, 2], **kwargs)


def evo_resnet34(**kwargs):
    """Constructs a ResNet-34 model.

    """
    return EvoResNet([3, 4, 6, 3], **kwargs)


def evo_resnet50(**kwargs):
    """Constructs a ResNet-50 model.

    """
    return EvoResNet([3, 4, 6, 3], **kwargs)


def evo_resnet101(**kwargs):
    """Constructs a ResNet-101 model.

    """
    return EvoResNet([3, 4, 23, 3], **kwargs)


def evo_resnet152(**kwargs):
    """Constructs a ResNet-152 model.

    """
    return EvoResNet([3, 8, 36, 3], **kwargs)


def evo_resnext50_32x4d(**kwargs):
    """Constructs a ResNeXt-50 32x4d model.

    """
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 4
    return EvoResNet([3, 4, 6, 3], **kwargs)


def evo_resnext101_32x8d(**kwargs):
    """Constructs a ResNeXt-101 32x8d model.

    """
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 8
    return EvoResNet([3, 4, 23, 3], **kwargs)
