# -*- coding: utf-8 -*-
from models import MobileNetV3_Large, MobileNetV3_Small, MobileNetV1, MobileNetV2
from torchtoolbox.tools import summary
import torch

model = MobileNetV3_Large()
# model = MobileNetV1()
# model = MobileNetV2()
# model = MobileNetV3_Small()
summary(model, torch.rand(1, 3, 224, 224))
