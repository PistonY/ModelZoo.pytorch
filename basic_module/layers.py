__all__ = ['fs_bn', 'fs_relu']
import torch
from torch import nn


class fs_bn(nn.Module):
    def __init__(self, channels, alpha):
        super().__init__()
        h_out = int((1 - alpha) * channels)
        l_out = int(alpha * channels)
        self.h_bn = nn.BatchNorm2d(h_out)
        self.l_bn = nn.BatchNorm2d(l_out) if alpha != 0 else None

    def forward(self, x_h, x_l=None):
        y_h = self.h_bn(x_h)
        y_l = self.l_bn(x_l) if x_l is not None else None
        return y_h, y_l


class fs_relu(nn.Module):
    def __init__(self):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x_h, x_l=None):
        y_h = self.relu(x_h)
        y_l = self.relu(x_l) if x_l is not None else None
        return y_h, y_l


class se(nn.Module):
    def __init__(self, channels, reduction=4):
        super().__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, input):
        y = self.se(input)
        return input * y


class cbam(nn.Module):
    def __init__(self, channel, reduction=4, k=3):
        super().__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.maxpool = nn.AdaptiveMaxPool2d(1)
        self.cat = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1),
        )
        assert k in (3, 7)
        padding = 3 if k == 7 else 1
        self.sat = nn.Sequential(
            nn.Conv2d(2, 1, k, 1, padding, bias=False),
            nn.BatchNorm2d(1),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # ChannelAttention
        avg_out = self.cat(self.avgpool(x))
        max_out = self.cat(self.maxpool(x))
        out = self.sigmoid(avg_out + max_out)
        x = x * out
        # SpatialAttention
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.sat(out)
        out = self.sigmoid(out)
        return x * out
