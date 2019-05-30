from models.resnet import *
import torch

model = resnet50v2(0)
dt = torch.randn(1, 3, 224, 224)

out = model(dt)
print(out.size())
