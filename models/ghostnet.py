__all__ = ['GhostNet']

import math
import torch
from torch import nn
from torchtoolbox.nn import Activation


def make_divisible(v, divisible_by, min_value=None):
    """
    This function is taken from the original tf repo.
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    """
    if min_value is None:
        min_value = divisible_by
    new_v = max(min_value, int(v + divisible_by / 2) // divisible_by * divisible_by)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisible_by
    return new_v


class SE(nn.Module):
    def __init__(self, in_c, reduction_ratio=0.25):
        super(SE, self).__init__()
        reducation_c = make_divisible(in_c * reduction_ratio, 4)
        self.block = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_c, reducation_c, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(reducation_c, in_c, kernel_size=1, bias=True),
            nn.Hardsigmoid()
        )

    def forward(self, x):
        return x * self.block(x)


class GhostModule(nn.Module):
    def __init__(self, in_c, out_c, kernel_size=1, ratio=2, dw_size=3, stride=1, act=True, act_type='relu'):
        super(GhostModule, self).__init__()
        if ratio != 2:
            print("Please change output channels manually.")
        init_c = math.ceil(out_c / ratio)
        new_c = init_c * (ratio - 1)

        self.primary_conv = nn.Sequential(
            nn.Conv2d(in_c, init_c, kernel_size, stride, kernel_size // 2, bias=False),
            nn.BatchNorm2d(init_c),
            Activation(act_type) if act else nn.Identity()
        )

        self.cheap_operation = nn.Sequential(
            nn.Conv2d(init_c, new_c, dw_size, 1, dw_size // 2, groups=init_c, bias=False),
            nn.BatchNorm2d(new_c),
            Activation(act_type) if act else nn.Identity()
        )

    def forward(self, x):
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        out = torch.cat([x1, x2], dim=1)
        # if ratio != 2, you need return out[:,:out_c,:,:]
        return out


class GhostBottleneck(nn.Module):
    def __init__(self, in_c, mid_c, out_c, dw_kernel_size=3, stride=1, se_ratio=None, act_type='relu'):
        super(GhostBottleneck, self).__init__()
        self.ghost1 = GhostModule(in_c, mid_c, act=True, act_type=act_type)
        if stride > 1:
            self.dw_conv = nn.Sequential(
                nn.Conv2d(mid_c, mid_c, dw_kernel_size, stride,
                          dw_kernel_size // 2, groups=mid_c, bias=False),
                nn.BatchNorm2d(mid_c)
            )
        else:
            self.dw_conv = nn.Identity()
        self.se = SE(mid_c, reduction_ratio=se_ratio) if se_ratio is not None else nn.Identity()
        self.ghost2 = GhostModule(mid_c, out_c, act=False, act_type=act_type)

        if in_c == out_c and stride == 1:
            self.shortcut = nn.Identity()
        else:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_c, in_c, dw_kernel_size, stride,
                          dw_kernel_size // 2, groups=in_c, bias=False),
                nn.BatchNorm2d(in_c),
                nn.Conv2d(in_c, out_c, 1, bias=False),
                nn.BatchNorm2d(out_c)
            )

    def forward(self, x):
        residual = self.shortcut(x)
        x = self.ghost1(x)
        x = self.dw_conv(x)
        x = self.se(x)
        x = self.ghost2(x)
        return x + residual


class Stem(nn.Module):
    def __init__(self, out_c, act_type='relu'):
        super(Stem, self).__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, out_c, 3, 2, 1, bias=False),
            nn.BatchNorm2d(out_c),
            Activation(act_type)
        )

    def forward(self, x):
        return self.stem(x)


class Head(nn.Module):
    def __init__(self, in_c, mid_c, out_c, dropout, act_type='relu'):
        super(Head, self).__init__()
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_c, mid_c, 1, bias=True),
            Activation(act_type),
            nn.Flatten(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(mid_c, out_c)
        )

    def forward(self, x):
        return self.head(x)


class GhostNet(nn.Module):
    def __init__(self, num_classes=1000, width=1.3, dropout=0):
        super(GhostNet, self).__init__()
        assert dropout >= 0, "Use = 0 to disable or > 0 to enable."
        self.width = width
        stem_c = make_divisible(16 * width, 4)
        self.stem = Stem(stem_c)
        self.stage = nn.Sequential(
            # stage1
            GhostBottleneck(stem_c, self.get_c(16), self.get_c(16), 3, 1),
            # stage2
            GhostBottleneck(self.get_c(16), self.get_c(48), self.get_c(24), 3, 2),
            GhostBottleneck(self.get_c(24), self.get_c(72), self.get_c(24), 3, 1),
            # stage3
            GhostBottleneck(self.get_c(24), self.get_c(72), self.get_c(40), 5, 2, 0.25),
            GhostBottleneck(self.get_c(40), self.get_c(120), self.get_c(40), 5, 1, 0.25),
            # stage4
            GhostBottleneck(self.get_c(40), self.get_c(240), self.get_c(80), 3, 2),
            GhostBottleneck(self.get_c(80), self.get_c(200), self.get_c(80), 3, 1),
            GhostBottleneck(self.get_c(80), self.get_c(184), self.get_c(80), 3, 1),
            GhostBottleneck(self.get_c(80), self.get_c(184), self.get_c(80), 3, 1),
            GhostBottleneck(self.get_c(80), self.get_c(480), self.get_c(112), 3, 1, 0.25),
            GhostBottleneck(self.get_c(112), self.get_c(672), self.get_c(112), 3, 1, 0.25),
            # stage5
            GhostBottleneck(self.get_c(112), self.get_c(672), self.get_c(160), 5, 2, 0.25),
            GhostBottleneck(self.get_c(160), self.get_c(960), self.get_c(160), 5, 1),
            GhostBottleneck(self.get_c(160), self.get_c(960), self.get_c(160), 5, 1, 0.25),
            GhostBottleneck(self.get_c(160), self.get_c(960), self.get_c(160), 5, 1),
            GhostBottleneck(self.get_c(160), self.get_c(960), self.get_c(160), 5, 1, 0.25),
            # conv-bn-act
            nn.Conv2d(self.get_c(160), self.get_c(960), 1, bias=False),
            nn.BatchNorm2d(self.get_c(960)),
            nn.ReLU(inplace=True),
        )

        self.head = Head(self.get_c(960), 1280, num_classes, dropout)

    def get_c(self, c):
        return make_divisible(c * self.width, 4)

    def forward(self, x):
        x = self.stem(x)
        x = self.stage(x)
        x = self.head(x)
        return x


class GhostNet600(nn.Module):
    def __init__(self, num_classes=1000, width=1.75, dropout=0.8):
        super(GhostNet600, self).__init__()
        assert dropout >= 0, "Use = 0 to disable or > 0 to enable."
        self.width = width
        stem_c = make_divisible(16 * width, 4)
        self.stem = Stem(stem_c, 'h_swish')
        self.stage = nn.Sequential(
            # stage1
            GhostBottleneck(stem_c, self.get_c(16), self.get_c(16), 3, 1, 0.1, 'h_swish'),
            # stage2
            GhostBottleneck(self.get_c(16), self.get_c(48), self.get_c(24), 3, 2, 0.1, 'h_swish'),
            GhostBottleneck(self.get_c(24), self.get_c(72), self.get_c(24), 3, 1, 0.1, 'h_swish'),
            # stage3
            GhostBottleneck(self.get_c(24), self.get_c(72), self.get_c(40), 5, 2, 0.1, 'h_swish'),
            GhostBottleneck(self.get_c(40), self.get_c(120), self.get_c(40), 3, 1, 0.1, 'h_swish'),
            GhostBottleneck(self.get_c(40), self.get_c(120), self.get_c(40), 3, 1, 0.1, 'h_swish'),
            # stage4
            GhostBottleneck(self.get_c(40), self.get_c(240), self.get_c(80), 3, 2, 0.1, 'h_swish'),
            GhostBottleneck(self.get_c(80), self.get_c(200), self.get_c(80), 3, 1, 0.1, 'h_swish'),
            GhostBottleneck(self.get_c(80), self.get_c(200), self.get_c(80), 3, 1, 0.1, 'h_swish'),
            GhostBottleneck(self.get_c(80), self.get_c(200), self.get_c(80), 3, 1, 0.1, 'h_swish'),
            GhostBottleneck(self.get_c(80), self.get_c(480), self.get_c(112), 3, 1, 0.1, 'h_swish'),
            GhostBottleneck(self.get_c(112), self.get_c(672), self.get_c(112), 3, 1, 0.1, 'h_swish'),
            GhostBottleneck(self.get_c(112), self.get_c(672), self.get_c(112), 3, 1, 0.1, 'h_swish'),
            # stage5
            GhostBottleneck(self.get_c(112), self.get_c(672), self.get_c(160), 5, 2, 0.1, 'h_swish'),
            GhostBottleneck(self.get_c(160), self.get_c(960), self.get_c(160), 3, 1, 0.1, 'h_swish'),
            GhostBottleneck(self.get_c(160), self.get_c(960), self.get_c(160), 3, 1, 0.1, 'h_swish'),
            GhostBottleneck(self.get_c(160), self.get_c(960), self.get_c(160), 3, 1, 0.1, 'h_swish'),
            GhostBottleneck(self.get_c(160), self.get_c(960), self.get_c(160), 3, 1, 0.1, 'h_swish'),
            GhostBottleneck(self.get_c(160), self.get_c(960), self.get_c(160), 3, 1, 0.1, 'h_swish'),
            # conv-bn-act
            nn.Conv2d(self.get_c(160), self.get_c(960), 1, bias=False),
            nn.BatchNorm2d(self.get_c(960)),
            Activation('h_swish'),
        )

        self.head = Head(self.get_c(960), 1400, num_classes, dropout, 'h_swish')

    def get_c(self, c):
        return make_divisible(c * self.width, 4)

    def forward(self, x):
        x = self.stem(x)
        x = self.stage(x)
        x = self.head(x)
        return x


class TinyGhostNet(nn.Module):
    def __init__(self, width_coeff, depth_coeff, depth_div=4,
                 min_depth=None, num_classes=1000, dropout=0):
        super().__init__()
        assert dropout >= 0, "Use = 0 to disable or > 0 to enable."
        min_depth = min_depth or depth_div

        def renew_ch(x):
            if not width_coeff:
                return x

            new_x = max(min_depth, int(x + depth_div / 2) // depth_div * depth_div)
            if new_x < 0.9 * new_x:
                new_x += depth_div
            return new_x

        def renew_repeat(x):
            return int(math.ceil(x * depth_coeff))

        self.stem = Stem(renew_ch(28))
        self.stage = nn.Sequential(
            # stage1
            self._make_layer(renew_ch(28), renew_ch(28), 1, 3, 1, renew_repeat(1), 0.1),
            # stage2
            self._make_layer(renew_ch(28), renew_ch(44), 3, 3, 2, renew_repeat(1), 0.1),
            self._make_layer(renew_ch(44), renew_ch(44), 3, 3, 1, renew_repeat(1), 0.1),
            # stage3
            self._make_layer(renew_ch(44), renew_ch(72), 3, 3, 2, renew_repeat(1), 0.1),
            self._make_layer(renew_ch(72), renew_ch(72), 3, 3, 1, renew_repeat(2), 0.1),
            # stage4
            self._make_layer(renew_ch(72), renew_ch(140), 6, 3, 2, renew_repeat(1), 0.1),
            self._make_layer(renew_ch(140), renew_ch(140), 2.5, 3, 1, renew_repeat(3), 0.1),
            self._make_layer(renew_ch(140), renew_ch(196), 6, 3, 1, renew_repeat(3), 0.1),
            # stage5
            self._make_layer(renew_ch(196), renew_ch(280), 6, 3, 2, renew_repeat(1), 0.1),
            self._make_layer(renew_ch(280), renew_ch(280), 6, 3, 1, renew_repeat(5), 0.1),
            nn.Conv2d(renew_ch(280), renew_ch(1680), 1, bias=False),
            nn.BatchNorm2d(renew_ch(1680)),
            nn.ReLU(inplace=True)
        )
        self.head = Head(renew_ch(1680), renew_ch(1400), num_classes, dropout)

    def _make_layer(self, in_c, out_c, expand, kernel_size, stride, repeats, se_ratio):
        layers = []

        layers.append(GhostBottleneck(in_c, int(in_c * expand), out_c, kernel_size, stride, se_ratio))
        for _ in range(repeats - 1):
            layers.append(GhostBottleneck(out_c, int(out_c * expand), out_c, kernel_size, 1, se_ratio))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.stem(x)
        x = self.stage(x)
        x = self.head(x)
        return x


def GhostNetA(num_classes=1000, dropout=0, **kwargs):
    return TinyGhostNet(1., 1., num_classes=num_classes, dropout=dropout, **kwargs)


if __name__ == '__main__':
    from torchtoolbox.tools import summary

    model = GhostNet600()
    x = torch.rand(size=(1, 3, 224, 224))
    summary(model, x)
