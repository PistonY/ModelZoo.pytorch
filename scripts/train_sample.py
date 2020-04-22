# -*- coding: utf-8 -*-
# @Author  : DevinYang(pistonyang@gmail.com)

import argparse, time, logging, os
import models
import torch
import warnings
import apex
from scripts.utils import get_model, set_model

from torchtoolbox import metric
from torchtoolbox.nn import LabelSmoothingLoss
from torchtoolbox.optimizer import CosineWarmupLr, Lookahead
from torchtoolbox.nn.init import KaimingInitializer
from torchtoolbox.tools import no_decay_bias, \
    mixup_data, mixup_criterion, check_dir, summary

from torchvision import transforms
from torchvision.datasets import CIFAR10, CIFAR100, MNIST, FashionMNIST, ImageFolder
from torch.utils.data import DataLoader
from torch import nn
from torch import optim
from apex import amp

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

parser = argparse.ArgumentParser(description='Train a model on ImageNet.')
parser.add_argument('--data-path', type=str, required=True,
                    help='training and validation dataset.')
parser.add_argument('--dataset', type=str, default='mnist',
                    help='Which dataset to train, default is mnist')
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
parser.add_argument('--lr', type=float, default=0,
                    help='learning rate. default is 0.')
parser.add_argument('--momentum', type=float, default=0.9,
                    help='momentum value for optimizer, default is 0.9.')
parser.add_argument('--wd', type=float, default=0.0001,
                    help='weight decay rate. default is 0.0001.')
parser.add_argument('--dropout', type=float, default=0.,
                    help='model dropout rate.')
parser.add_argument('--sync-bn', action='store_true',
                    help='use Apex Sync-BN.')
parser.add_argument('--lookahead', action='store_true',
                    help='use lookahead optimizer.')
parser.add_argument('--warmup-lr', type=float, default=0.0,
                    help='starting warmup learning rate. default is 0.0.')
parser.add_argument('--warmup-epochs', type=int, default=0,
                    help='number of warmup epochs.')
parser.add_argument('--model', type=str, required=True,
                    help='type of model to use. see vision_model for options.')
parser.add_argument('--alpha', type=float, default=0,
                    help='model param.')
parser.add_argument('--input-size', type=int, default=32,
                    help='size of the input image size. default is 224')
parser.add_argument('--padding', type=int, default=4,
                    help='pad input image')
parser.add_argument('--norm-layer', type=str, default='',
                    help='Norm layer to use.')
parser.add_argument('--activation', type=str, default='',
                    help='activation to use.')
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
parser.add_argument('--logging-file', type=str, default='train_sample.log',
                    help='name of training log file')
parser.add_argument('--resume-epoch', type=int, default=0,
                    help='epoch to resume training from.')
parser.add_argument('--resume-param', type=str, default='',
                    help='resume training param path.')
parser.add_argument("--local_rank", default=0, type=int)
args = parser.parse_args()


def get_dataset(name):
    path = args.data_path
    download = not os.path.exists(path) or not os.listdir(path)
    if name == 'cifar10':
        train_ = CIFAR10(path, train=True, transform=train_transform, download=download)
        val_ = CIFAR10(path, train=False, transform=val_transform, download=download)
    elif name == 'cifar100':
        train_ = CIFAR100(path, train=True, transform=train_transform, download=download)
        val_ = CIFAR100(path, train=False, transform=val_transform, download=download)
    elif name == 'mnist':
        train_ = MNIST(path, train=True, transform=train_transform, download=download)
        val_ = MNIST(path, train=False, transform=val_transform, download=download)
    elif name == 'fashion_mnist':
        train_ = FashionMNIST(path, train=True, transform=train_transform, download=download)
        val_ = FashionMNIST(path, train=False, transform=val_transform, download=download)
    else:
        train_ = ImageFolder(os.path.join(path, 'train'), transform=train_transform)
        val_ = ImageFolder(os.path.join(path, 'val'), transform=val_transform)
    return train_, val_


filehandler = logging.FileHandler(args.logging_file)
streamhandler = logging.StreamHandler()

logger = logging.getLogger('')
logger.setLevel(logging.INFO)
logger.addHandler(filehandler)
logger.addHandler(streamhandler)

logger.info(args)
warnings.filterwarnings("ignore", "(Possibly )?corrupt EXIF data", UserWarning)

torch.backends.cudnn.benchmark = True

train_transform = transforms.Compose([
    transforms.RandomCrop(args.input_size, padding=args.padding),
    # Cutout(),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])

val_transform = transforms.ToTensor()
train_set, val_set = get_dataset(args.dataset)

classes = len(train_set.classes)
num_training_samples = len(train_set)

check_dir(args.save_dir)
assert torch.cuda.is_available(), \
    "Please don't waste of your time,it's impossible to train on CPU."

device = torch.device("cuda:0")
device_ids = args.devices.strip().split(',')
device_ids = [int(device) for device in device_ids]

dtype = args.dtype
epochs = args.epochs
resume_epoch = args.resume_epoch
num_workers = args.num_workers
initializer = KaimingInitializer()
batch_size = args.batch_size * len(device_ids)
batches_pre_epoch = num_training_samples // batch_size
lr = 0.1 * (args.batch_size // 32) if args.lr == 0 else args.lr

train_data = DataLoader(train_set, batch_size, False, pin_memory=True, num_workers=num_workers, drop_last=True)
val_data = DataLoader(val_set, batch_size, False, pin_memory=True, num_workers=num_workers, drop_last=False)

model_setting = set_model(args.dropout, args.norm_layer, args.activation)

try:
    model = get_model(models, args.model, alpha=args.alpha, small_input=True, **model_setting)
except TypeError:
    model = get_model(models, args.model, small_input=True, **model_setting)

summary(model, torch.rand((1, 3, args.input_size, args.input_size)))
model.apply(initializer)
model.to(device)
parameters = model.parameters() if not args.no_wd else no_decay_bias(model)
optimizer = optim.SGD(parameters, lr=lr, momentum=args.momentum,
                      weight_decay=args.wd, nesterov=True)

if args.sync_bn:
    logger.info('Use Apex Synced BN.')
    model = apex.parallel.convert_syncbn_model(model)

if dtype == 'float16':
    logger.info('Train with FP16.')
    model, optimizer = amp.initialize(model, optimizer, opt_level='O1')

if args.lookahead:
    logger.info('Use lookahead optimizer.')
    optimizer = Lookahead(optimizer)

model = nn.DataParallel(model)
lr_scheduler = CosineWarmupLr(optimizer, batches_pre_epoch, epochs,
                              base_lr=args.lr, warmup_epochs=args.warmup_epochs)
if resume_epoch > 0:
    checkpoint = torch.load(args.resume_param)
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    amp.load_state_dict(checkpoint['amp'])
    lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
    print("Finish loading resume param.")

top1_acc = metric.Accuracy(name='Top1 Accuracy')
top5_acc = metric.TopKAccuracy(top=5, name='Top5 Accuracy')
loss_record = metric.NumericalCost(name='Loss')

Loss = nn.CrossEntropyLoss() if not args.label_smoothing else \
    LabelSmoothingLoss(classes, smoothing=0.1)


@torch.no_grad()
def test(epoch=0, save_status=False):
    top1_acc.reset()
    top5_acc.reset()
    loss_record.reset()
    model.eval()
    for data, labels in val_data:
        data = data.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        outputs = model(data)
        losses = Loss(outputs, labels)

        top1_acc.update(outputs, labels)
        top5_acc.update(outputs, labels)
        loss_record.update(losses)

    test_msg = 'Test Epoch {}: {}:{:.5}, {}:{:.5}, {}:{:.5}\n'.format(
        epoch, top1_acc.name, top1_acc.get(), top5_acc.name, top5_acc.get(),
        loss_record.name, loss_record.get())
    logger.info(test_msg)
    if save_status:
        checkpoint = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'amp': amp.state_dict(),
            'lr_scheduler': lr_scheduler.state_dict(),
        }
        torch.save(checkpoint, '{}/{}_{}_{:.5}.pt'.format(
            args.save_dir, args.model, epoch, top1_acc.get()))


def train():
    for epoch in range(resume_epoch, epochs):
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
            top1_acc.update(outputs, labels)
            loss_record.update(loss)

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
    for epoch in range(resume_epoch, epochs):
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

            loss_record.update(loss)
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
