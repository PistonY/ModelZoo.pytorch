# -*- coding: utf-8 -*-
# @Author  : DevinYang(pistonyang@gmail.com)

import argparse, time, logging
import models
import torch
import warnings
import apex
from torchtoolbox import metric
from torchtoolbox.nn import LabelSmoothingLoss
from torchtoolbox.nn.init import KaimingInitializer
from torchtoolbox.tools import split_weights, CosineWarmupLr, \
    mixup_data, mixup_criterion
from torch.nn import functional as F
from torchvision import transforms
from torchvision.datasets import ImageNet
from torch.utils.data import DataLoader
from torch import nn
from torch import optim
from apex import amp

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

parser = argparse.ArgumentParser(description='Train a model on ImageNet.')
parser.add_argument('--data-path', type=str, required=True,
                    help='training and validation dataset.')
parser.add_argument('--batch-size', type=int, default=32,
                    help='training batch size per device (CPU/GPU).')
parser.add_argument('--dtype', type=str, default='float32',
                    help='data type for training. default is float32')
parser.add_argument('--devices', type=str, default='0',
                    help='gpus to use.')
parser.add_argument('-j', '--num-data-workers', dest='num_workers', default=4, type=int,
                    help='number of preprocessing workers')
parser.add_argument('--epochs', type=int, default=1,
                    help='number of training epochs.')
parser.add_argument('--lr', type=float, default=0.1,
                    help='learning rate. default is 0.1.')
parser.add_argument('--momentum', type=float, default=0.9,
                    help='momentum value for optimizer, default is 0.9.')
parser.add_argument('--wd', type=float, default=0.0001,
                    help='weight decay rate. default is 0.0001.')
parser.add_argument('--sync-bn', action='store_true',
                    help='use Apex Sync-BN.')
# parser.add_argument('--lr-mode', type=str, default='step',
#                     help='learning rate scheduler mode. options are step, poly and cosine.')
# parser.add_argument('--lr-decay', type=float, default=0.1,
#                     help='decay rate of learning rate. default is 0.1.')
# parser.add_argument('--lr-decay-period', type=int, default=0,
#                     help='interval for periodic learning rate decays. default is 0 to disable.')
# parser.add_argument('--lr-decay-epoch', type=str, default='40,60',
#                     help='epochs at which learning rate decays. default is 40,60.')
parser.add_argument('--warmup-lr', type=float, default=0.0,
                    help='starting warmup learning rate. default is 0.0.')
parser.add_argument('--warmup-epochs', type=int, default=0,
                    help='number of warmup epochs.')
parser.add_argument('--model', type=str, required=True,
                    help='type of model to use. see vision_model for options.')
parser.add_argument('--alpha', type=float, default=0,
                    help='model param.')
parser.add_argument('--input-size', type=int, default=224,
                    help='size of the input image size. default is 224')
parser.add_argument('--crop-ratio', type=float, default=0.875,
                    help='Crop ratio during validation. default is 0.875')
parser.add_argument('--mixup', action='store_true',
                    help='whether train the model with mix-up. default is false.')
parser.add_argument('--mixup-alpha', type=float, default=0.2,
                    help='beta distribution parameter for mixup sampling, default is 0.2.')
parser.add_argument('--mixup-off-epoch', type=int, default=0,
                    help='how many epochs to train without mixup, default is 0.')
parser.add_argument('--label-smoothing', action='store_true',
                    help='use label smoothing or not in training. default is false.')
parser.add_argument('--no-wd', action='store_true',
                    help='whether to remove weight decay on bias, and beta/gamma for batchnorm layers.')
parser.add_argument('--save-dir', type=str, default='params',
                    help='directory of saved models')
parser.add_argument('--log-interval', type=int, default=50,
                    help='Number of batches to wait before logging.')
parser.add_argument('--logging-file', type=str, default='train_imagenet.log',
                    help='name of training log file')
args = parser.parse_args()

filehandler = logging.FileHandler(args.logging_file)
streamhandler = logging.StreamHandler()

logger = logging.getLogger('')
logger.setLevel(logging.INFO)
logger.addHandler(filehandler)
logger.addHandler(streamhandler)

logger.info(args)
warnings.filterwarnings("ignore", "(Possibly )?corrupt EXIF data", UserWarning)

torch.backends.cudnn.benchmark = True


def get_model(name, **kwargs):
    return models.__dict__[name](**kwargs)


classes = 1000
num_training_samples = 1281167

assert torch.cuda.is_available(), \
    "Please don't waste of your time,it's impossible to train on CPU."

device = torch.device("cuda:0")
device_ids = args.devices.strip().split(',')
device_ids = [int(device) for device in device_ids]

batch_size = args.batch_size * len(device_ids)
epochs = args.epochs
batches_pre_epoch = num_training_samples // batch_size
num_workers = args.num_workers

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

_train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(0.4, 0.4, 0.4),
    # transforms.RandomRotation(15),
    transforms.ToTensor(),
    normalize,
])

_val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    normalize,
])

train_data = DataLoader(ImageNet(args.data_path, split='train', transform=_train_transform),
                        batch_size, True, pin_memory=True, num_workers=num_workers, drop_last=True)
val_data = DataLoader(ImageNet(args.data_path, split='val', transform=_val_transform),
                      batch_size, False, pin_memory=True, num_workers=num_workers, drop_last=False)

try:
    model = get_model(args.model, alpha=args.alpha)
except TypeError:
    model = get_model(args.model)

if args.sync_bn:
    logger.info('Using Apex Synced BN.')
    model = apex.parallel.convert_syncbn_model(model)

parameters = model.parameters() if not args.no_wd else split_weights(model)
optimizer = optim.SGD(parameters, lr=args.lr, momentum=args.momentum,
                      weight_decay=args.wd, nesterov=True)
KaimingInitializer(model)
model.to(device)
dtype = args.dtype

if dtype == 'float16':
    logger.info('Train with FP16.')
    model, optimizer = amp.initialize(model, optimizer, opt_level='O1')

model = nn.DataParallel(model, device_ids=device_ids)

lr_scheduler = CosineWarmupLr(optimizer, batches_pre_epoch, epochs,
                              base_lr=args.lr, warmup_epochs=args.warmup_epochs)

top1_acc = metric.Accuracy(name='Top1 Accuracy')
top5_acc = metric.TopKAccuracy(top=5, name='Top5 Accuracy')
loss_record = metric.NumericalCost(name='Loss')

Loss = nn.CrossEntropyLoss() if not args.label_smoothing else \
    LabelSmoothingLoss(classes, smoothing=0.1)


@torch.no_grad()
def test(epoch=0, save_status=True):
    top1_acc.reset()
    top5_acc.reset()
    loss_record.reset()
    model.eval()
    for data, labels in val_data:
        data = data.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        outputs = model(data)
        losses = Loss(outputs, labels)

        top1_acc.step(outputs, labels)
        top5_acc.step(outputs, labels)
        loss_record.step(losses)

    test_msg = 'Test Epoch {}: {}:{:.5}, {}:{:.5}, {}:{:.5}\n'.format(
        epoch, top1_acc.name, top1_acc.get(), top5_acc.name, top5_acc.get(),
        loss_record.name, loss_record.get())
    logger.info(test_msg)
    if save_status:
        torch.save(model.state_dict(), '{}/{}_{}_{:.5}.pkl'.format(
            args.save_dir, args.model, epoch, top1_acc.get()))


def train():
    for epoch in range(epochs):
        top1_acc.reset()
        loss_record.reset()
        tic = time.time()

        model.train()
        for i, (data, labels) in enumerate(train_data):
            data = data.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad()
            outputs = model(data)
            loss = Loss(outputs, labels)

            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
            optimizer.step()

            lr_scheduler.step()
            top1_acc.step(outputs, labels)
            loss_record.step(loss)

            if i % args.log_interval == 0 and i != 0:
                logger.info('Epoch {}, Iter {}, {}:{:.5}, {}:{:.5}, {} samples/s. lr: {:.5}.'.format(
                    epoch, i, top1_acc.name, top1_acc.get(),
                    loss_record.name, loss_record.get(),
                    int((i * batch_size) // (time.time() - tic)),
                    lr_scheduler.learning_rate
                ))

        train_speed = int(num_training_samples // (time.time() - tic))
        epoch_msg = 'Train Epoch {}: {}:{:.5}, {}:{:.5}, {} samples/s.'.format(
            epoch, top1_acc.name, top1_acc.get(), loss_record.name, loss_record.get(), train_speed)
        logger.info(epoch_msg)
        test(epoch)


def train_mixup():
    mixup_off_epoch = epochs if args.mixup_off_epoch == 0 else args.mixup_off_epoch
    for epoch in range(epochs):
        loss_record.reset()
        alpha = args.mixup_alpha if epoch < mixup_off_epoch else 0
        tic = time.time()

        model.train()
        for i, (data, labels) in enumerate(train_data):
            data = data.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            data, labels_a, labels_b, lam = mixup_data(data, labels, alpha)
            optimizer.zero_grad()
            outputs = model(data)
            loss = mixup_criterion(Loss, outputs, labels_a, labels_b, lam)

            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
            optimizer.step()

            loss_record.step(loss)
            lr_scheduler.step()

            if i % args.log_interval == 0 and i != 0:
                logger.info('Epoch {}, Iter {}, {}:{:.5}, {} samples/s.'.format(
                    epoch, i, loss_record.name, loss_record.get(),
                    int((i * batch_size) // (time.time() - tic))
                ))

        train_speed = int(num_training_samples // (time.time() - tic))
        train_msg = 'Train Epoch {}: {}:{:.5}, {} samples/s, lr:{:.5}'.format(
            epoch, loss_record.name, loss_record.get(),
            train_speed, lr_scheduler.learning_rate)
        logger.info(train_msg)
        test(epoch)


if __name__ == '__main__':
    if args.mixup:
        logger.info('Train using Mixup.')
        train_mixup()
    else:
        train()
