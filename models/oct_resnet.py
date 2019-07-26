__all__ = ['oct_resnet50', 'oct_resnet50v2']

from oct_module import *
from utils import *
from torch import nn


def check_status(alpha_in, alpha_out):
    alpha_in = alpha_out if alpha_in == 0 else alpha_in
    alpha_in = 0 if alpha_out == 0 else alpha_in
    return alpha_in, alpha_out


class OctBottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, alpha_in, alpha_out,
                 stride=1, groups=1, base_width=64):
        super(OctBottleneck, self).__init__()
        width = int(planes * (base_width / 64.)) * groups
        if stride != 1 or inplanes != planes * self.expansion:
            self.downsample = AdaptiveSequential(
                OctaveConv(inplanes, planes * self.expansion, alpha_in, alpha_out,
                           1, stride=stride, bias=False),
                fs_bn(planes * self.expansion, alpha_out)
            )
        else:
            self.downsample = None

        self.conv1 = OctaveConv(inplanes, width, alpha_in, alpha_out, 1, bias=False)
        self.bn1 = fs_bn(width, alpha_out)
        alpha_in, alpha_out = check_status(alpha_in, alpha_out)
        self.conv2 = OctaveConv(width, width, alpha_in, alpha_out, 3, 1, stride,
                                groups, False)
        self.bn2 = fs_bn(width, alpha_out)
        self.conv3 = OctaveConv(width, planes * self.expansion, alpha_in, alpha_out,
                                1, bias=False)
        self.bn3 = fs_bn(planes * self.expansion, alpha_out)
        self.relu = fs_relu()

    def forward(self, x_h, x_l=None):
        r_h, r_l = x_h, x_l
        x_h, x_l = self.conv1(x_h, x_l)
        x_h, x_l = self.bn1(x_h, x_l)
        x_h, x_l = self.relu(x_h, x_l)

        x_h, x_l = self.conv2(x_h, x_l)
        x_h, x_l = self.bn2(x_h, x_l)
        x_h, x_l = self.relu(x_h, x_l)

        x_h, x_l = self.conv3(x_h, x_l)
        x_h, x_l = self.bn3(x_h, x_l)

        if self.downsample:
            r_h, r_l = self.downsample(r_h, r_l)
        y_h, y_l = self.relu(x_h + r_h, None if x_l is None and r_l is None else x_l + r_l)
        return y_h, y_l


class OctBottleneckV2(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, alpha_in, alpha_out, stride=1,
                 groups=1, base_width=64):
        super().__init__()
        width = int(planes * (base_width / 64.)) * groups
        if stride != 1 or inplanes != planes * self.expansion:
            self.downsample = AdaptiveSequential(
                fs_bn(inplanes, alpha_in),
                fs_relu(),
                OctaveConv(inplanes, planes * self.expansion, alpha_in, alpha_out,
                           1, stride=stride, bias=False),
            )
        else:
            self.downsample = None
        self.bn1 = fs_bn(inplanes, alpha_in)
        self.conv1 = OctaveConv(inplanes, width, alpha_in, alpha_out, 1, bias=False)
        alpha_in, alpha_out = check_status(alpha_in, alpha_out)
        self.bn2 = fs_bn(width, alpha_in)
        self.conv2 = OctaveConv(width, width, alpha_in, alpha_out, 3, 1,
                                stride, groups, False)
        self.bn3 = fs_bn(width, alpha_in)
        self.conv3 = OctaveConv(width, planes * self.expansion, alpha_in, alpha_in,
                                1, bias=False)
        self.relu = fs_relu()

    def forward(self, x_h, x_l=None):
        r_h, r_l = x_h, x_l
        x_h, x_l = self.bn1(x_h, x_l)
        x_h, x_l = self.relu(x_h, x_l)
        if self.downsample:
            r_h, r_l = self.downsample(x_h, x_l)
        x_h, x_l = self.conv1(x_h, x_l)

        x_h, x_l = self.bn2(x_h, x_l)
        x_h, x_l = self.relu(x_h, x_l)
        x_h, x_l = self.conv2(x_h, x_l)

        x_h, x_l = self.bn3(x_h, x_l)
        x_h, x_l = self.relu(x_h, x_l)
        x_h, x_l = self.conv3(x_h, x_l)

        y_h, y_l = x_h + r_h, None if x_l is None and r_l is None else x_l + r_l
        return y_h, y_l


class OctResNet(nn.Module):
    def __init__(self, alpha, layers, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64):
        super(OctResNet, self).__init__()
        self.inplanes = 64
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(alpha, 64, layers[0], 1, 'start')
        self.layer2 = self._make_layer(alpha, 128, layers[1], 2)
        self.layer3 = self._make_layer(alpha, 256, layers[2], 2)
        self.layer4 = self._make_layer(alpha, 512, layers[3], 2, 'end')
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * OctBottleneck.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, OctBottleneck):
                    nn.init.constant_(m.bn3.weight, 0)

    def _make_layer(self, alpha, planes, blocks, stride=1, status='normal'):
        assert status in ('start', 'normal', 'end')
        layers = []
        layers.append(OctBottleneck(self.inplanes, planes,
                                    alpha if status != 'start' else 0,
                                    alpha if status != 'end' else 0,
                                    stride, self.groups, self.base_width))
        self.inplanes = planes * OctBottleneck.expansion
        alpha = 0 if status == 'end' else alpha
        for _ in range(1, blocks):
            layers.append(OctBottleneck(self.inplanes, planes, alpha, alpha, 1,
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
        x, _ = self.layer4(x_h, x_l)

        x = self.avgpool(x)
        x = x.reshape(x.size(0), -1)
        x = self.fc(x)

        return x


class OctResNetV2(nn.Module):
    def __init__(self, alpha, layers, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64):
        super().__init__()
        self.inplanes = 64
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(alpha, 64, layers[0], 1, 'start')
        self.layer2 = self._make_layer(alpha, 128, layers[1], 2)
        self.layer3 = self._make_layer(alpha, 256, layers[2], 2)
        self.layer4 = self._make_layer(alpha, 512, layers[3], 2, 'end')
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * OctBottleneck.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, OctBottleneckV2):
                    nn.init.constant_(m.bn3.weight, 0)

    def _make_layer(self, alpha, planes, blocks, stride=1, status='normal'):
        assert status in ('start', 'normal', 'end')
        layers = []
        layers.append(OctBottleneckV2(self.inplanes, planes,
                                      alpha if status != 'start' else 0,
                                      alpha if status != 'end' else 0,
                                      stride, self.groups, self.base_width))
        self.inplanes = planes * OctBottleneckV2.expansion
        alpha = 0 if status == 'end' else alpha
        for _ in range(1, blocks):
            layers.append(OctBottleneckV2(self.inplanes, planes, alpha, alpha, 1,
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
        x, _ = self.layer4(x_h, x_l)

        x = self.avgpool(x)
        x = x.reshape(x.size(0), -1)
        x = self.fc(x)

        return x


def oct_resnet50(alpha, **kwargs):
    """Constructs a OctResNet-50 model.

    Args:
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return OctResNet(alpha, [3, 4, 6, 3], **kwargs)


def oct_resnet50v2(alpha, **kwargs):
    """Constructs a OctResNet-50 model.

    Args:
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return OctResNetV2(alpha, [3, 4, 6, 3], **kwargs)
