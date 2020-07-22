# -*- coding: utf-8 -*-
# @Author  : DevinYang(pistonyang@gmail.com)

import argparse
import os
import models
import torch

from torchtoolbox import metric

from torchtoolbox.nn import SwitchNorm2d, Swish

from torchtoolbox.data import ImageLMDB

from torchvision import transforms
from torchvision.datasets import ImageNet
from torch.utils.data import DataLoader
from torch import nn
from tqdm import tqdm

parser = argparse.ArgumentParser(description='Train a model on ImageNet.')
parser.add_argument('--data-path', type=str, required=True,
                    help='training and validation dataset.')
parser.add_argument('--use-lmdb', action='store_true',
                    help='use LMDB dataset/format')
parser.add_argument('--batch-size', type=int, default=32,
                    help='training batch size per device (CPU/GPU).')
parser.add_argument('--dtype', type=str, default='float32',
                    help='data type for training. default is float32')
parser.add_argument('--devices', type=str, default='0',
                    help='gpus to use.')
parser.add_argument('-j', '--num-data-workers', dest='num_workers', default=4, type=int,
                    help='number of preprocessing workers')
parser.add_argument('--model', type=str, required=True,
                    help='type of model to use. see vision_model for options.')
parser.add_argument('--alpha', type=float, default=0,
                    help='model param.')
parser.add_argument('--input-size', type=int, default=224,
                    help='size of the input image size. default is 224')
parser.add_argument('--norm-layer', type=str, default='',
                    help='Norm layer to use.')
parser.add_argument('--activation', type=str, default='',
                    help='activation to use.')
parser.add_argument('--param-path', type=str, default='',
                    help='param used to test.')

args = parser.parse_args()


def get_model(name, **kwargs):
    return models.__dict__[name](**kwargs)


def set_model(drop_out, norm_layer, act):
    setting = {}
    if drop_out != 0:
        setting['dropout_rate'] = args.dropout
    if norm_layer != '':
        if args.norm_layer == 'switch':
            setting['norm_layer'] = SwitchNorm2d
        else:
            raise NotImplementedError
    if act != '':
        if args.activation == 'swish':
            setting['activation'] = Swish()
        elif args.activation == 'relu6':
            setting['activation'] = nn.ReLU6(inplace=True)
        else:
            raise NotImplementedError
    return setting


classes = 1000
num_training_samples = 1281167

assert torch.cuda.is_available()
device = torch.device("cuda:0")
device_ids = args.devices.strip().split(',')
device_ids = [int(device) for device in device_ids]

dtype = args.dtype
num_workers = args.num_workers
batch_size = args.batch_size * len(device_ids)
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    normalize,
])

if not args.use_lmdb:
    val_set = ImageNet(args.data_path, split='val', transform=val_transform)
else:
    val_set = ImageLMDB(os.path.join(args.data_path, 'val.lmdb'), transform=val_transform)

val_data = DataLoader(val_set, batch_size, False, pin_memory=True, num_workers=num_workers, drop_last=False)

model_setting = set_model(0, args.norm_layer, args.activation)

try:
    model = get_model(args.model, alpha=args.alpha, **model_setting)
except TypeError:
    model = get_model(args.model, **model_setting)

model.to(device)
model = nn.DataParallel(model)

checkpoint = torch.load(args.param_path, map_location=device)
model.load_state_dict(checkpoint['model'])
print("Finish loading resume param.")

top1_acc = metric.Accuracy(name='Top1 Accuracy')
top5_acc = metric.TopKAccuracy(top=5, name='Top5 Accuracy')
loss_record = metric.NumericalCost(name='Loss')

Loss = nn.CrossEntropyLoss()


@torch.no_grad()
def test():
    top1_acc.reset()
    top5_acc.reset()
    loss_record.reset()
    model.eval()
    for data, labels in tqdm(val_data):
        data = data.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        outputs = model(data)
        losses = Loss(outputs, labels)

        top1_acc.update(outputs, labels)
        top5_acc.update(outputs, labels)
        loss_record.update(losses)

    test_msg = 'Test: {}:{:.5}, {}:{:.5}, {}:{:.5}\n'.format(
        top1_acc.name, top1_acc.get(), top5_acc.name, top5_acc.get(),
        loss_record.name, loss_record.get())
    print(test_msg)


if __name__ == '__main__':
    test()
