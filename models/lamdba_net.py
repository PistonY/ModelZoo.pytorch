import torch
import torch.nn.functional as F
from torch import nn, einsum
from einops import rearrange
from torchtoolbox.nn import Activation


# helpers functions

def exists(val):
    return val is not None


def default(val, d):
    return val if exists(val) else d


# lambda layer
def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class LambdaLayer(nn.Module):
    def __init__(self, dim, *, dim_k, n=None, r=None, heads=4, dim_out=None, dim_u=1):
        super().__init__()
        dim_out = default(dim_out, dim)
        self.u = dim_u  # intra-depth dimension
        self.heads = heads

        assert (dim_out % heads) == 0, 'values dimension must be divisible by number of heads for multi-head query'
        dim_v = dim_out // heads

        self.to_q = nn.Conv2d(dim, dim_k * heads, 1, bias=False)
        self.to_k = nn.Conv2d(dim, dim_k * dim_u, 1, bias=False)
        self.to_v = nn.Conv2d(dim, dim_v * dim_u, 1, bias=False)

        self.norm_q = nn.BatchNorm2d(dim_k * heads)
        self.norm_v = nn.BatchNorm2d(dim_v * dim_u)

        self.local_contexts = exists(r)
        if exists(r):
            assert (r % 2) == 1, 'Receptive kernel size should be odd'
            self.pos_conv = nn.Conv3d(dim_u, dim_k, (1, r, r), padding=(0, r // 2, r // 2))
        else:
            assert exists(n), 'You must specify the total sequence length (h x w)'
            self.pos_emb = nn.Parameter(torch.randn(n, n, dim_k, dim_u))

    def forward(self, x):
        b, c, hh, ww, u, h = *x.shape, self.u, self.heads

        q = self.to_q(x)
        k = self.to_k(x)
        v = self.to_v(x)

        q = self.norm_q(q)
        v = self.norm_v(v)

        q = rearrange(q, 'b (h k) hh ww -> b h k (hh ww)', h=h)
        k = rearrange(k, 'b (u k) hh ww -> b u k (hh ww)', u=u)
        v = rearrange(v, 'b (u v) hh ww -> b u v (hh ww)', u=u)

        k = k.softmax(dim=-1)

        λc = einsum('b u k m, b u v m -> b k v', k, v)
        Yc = einsum('b h k n, b k v -> b n h v', q, λc)

        if self.local_contexts:
            v = rearrange(v, 'b u v (hh ww) -> b u v hh ww', hh=hh, ww=ww)
            λp = self.pos_conv(v)
            Yp = einsum('b h k n, b k v n -> b n h v', q, λp.flatten(3))
        else:
            λp = einsum('n m k u, b u v m -> b n k v', self.pos_emb, v)
            Yp = einsum('b h k n, b n k v -> b n h v', q, λp)

        Y = Yc + Yp
        out = rearrange(Y, 'b (hh ww) h v -> b (h v) hh ww', hh=hh, ww=ww)
        return out.contiguous()


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()

        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)

        self.conv2 = LambdaLayer(planes, dim_k=16, r=15, heads=4, dim_u=1)
        self.bn2 = nn.BatchNorm2d(planes)
        self.pool = nn.AvgPool2d(3, 2, 1) if stride != 1 else nn.Identity()

        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.act = nn.ReLU(inplace=True)
        self.downsample = nn.Identity() if downsample is None else downsample

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.act(out)
        out = self.pool(out)

        out = self.conv3(out)
        out = self.bn3(out)

        identity = self.downsample(x)

        out += identity
        out = self.act(out)

        return out


class LambdaResnet(nn.Module):
    def __init__(self, layers, num_classes=1000, small_input=False):
        super(LambdaResnet, self).__init__()
        self.inplanes = 64
        if small_input:
            self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1,
                                   bias=False)
        else:
            self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                                   bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.act = nn.ReLU(inplace=True)
        if small_input:
            self.maxpool = nn.Identity()
        else:
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(64, layers[0])
        self.layer2 = self._make_layer(128, layers[1], stride=2)
        self.layer3 = self._make_layer(256, layers[2], stride=2)
        self.layer4 = self._make_layer(512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()

        self.fc = nn.Linear(512 * Bottleneck.expansion, num_classes)

    def _make_layer(self, planes, blocks, stride=1, ):

        downsample = None

        if stride != 1 or self.inplanes != planes * Bottleneck.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * Bottleneck.expansion, stride),
                nn.BatchNorm2d(planes * Bottleneck.expansion),
            )

        layers = []
        layers.append(Bottleneck(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * Bottleneck.expansion
        for _ in range(1, blocks):
            layers.append(Bottleneck(self.inplanes, planes))

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


def LambdaResnet18(**kwargs):
    """Constructs a ResNet-18 model.

    """
    return LambdaResnet([2, 2, 2, 2], **kwargs)


def LambdaResnet34(**kwargs):
    """Constructs a ResNet-34 model.

    """
    return LambdaResnet([3, 4, 6, 3], **kwargs)


def LambdaResnet50(**kwargs):
    """Constructs a ResNet-50 model.

    """
    return LambdaResnet([3, 4, 6, 3], **kwargs)


def LambdaResnet101(**kwargs):
    """Constructs a ResNet-101 model.

    """
    return LambdaResnet([3, 4, 23, 3], **kwargs)


def LambdaResnet152(**kwargs):
    """Constructs a ResNet-152 model.

    """
    return LambdaResnet([3, 8, 36, 3], **kwargs)
