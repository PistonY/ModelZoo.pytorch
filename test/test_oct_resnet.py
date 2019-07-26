from models.oct_resnet import *
import torch

model = oct_resnet50v2(0)
dt = torch.randn(1, 3, 224, 224)

out = model(dt)
print(out.size())
