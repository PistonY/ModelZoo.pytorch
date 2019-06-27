import torch
from oct_module.octconv import *

# test first layer
data1 = torch.randn(1, 64, 64, 64)
fo = OctaveConv(64, 64, 0, 0.125, 3, 1, 2)
out = fo(data1)
print(out[0].size(), out[1].size())

# test oct layer
oo = OctaveConv(64, 128, 0.125, 0.125, 3, 1, groups=128)
out = oo(out[0], out[1])
print(out[0].size(), out[1].size())

# test last layer
ol = OctaveConv(128, 128, 0.125, 0, 3, 1)
out = ol(out[0], out[1])
print(out[0].size())