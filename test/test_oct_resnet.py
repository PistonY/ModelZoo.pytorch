from models.oct_resnet import *
from torchtoolbox.tools.summary import summary
import torch

model = oct_resnet50v2(0)
dt = torch.randn(1, 3, 224, 224)

print(model)
# out = model(dt)
# summary(model, dt)
# print(out.size())
