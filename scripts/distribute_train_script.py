# -*- coding: utf-8 -*-
# @Author  : DevinYang(pistonyang@gmail.com)

import argparse
import time
import os
import models
import torch
import warnings

from scripts.utils import get_logger, get_model
from torchtoolbox import metric
from torchtoolbox.nn import LabelSmoothingLoss
from torchtoolbox.optimizer import CosineWarmupLr, Lookahead
from torchtoolbox.optimizer.sgd_gc import SGD_GC
from torchtoolbox.nn.init import KaimingInitializer
from torchtoolbox.tools import no_decay_bias, \
    mixup_data, mixup_criterion, check_dir, summary
from torchtoolbox.transform import Cutout

from torchvision.datasets import ImageNet
from torch import multiprocessing as mp
from torch.utils.data import DataLoader
from torch.utils.data import DistributedSampler
from torchvision import transforms
from torch import nn
from torch import optim
from torch.cuda.amp import autocast, GradScaler

from module.aa import ImageNetPolicy

# from module.dropblock import DropBlockScheduler

parser = argparse.ArgumentParser(description='Train a model on ImageNet.')
parser.add_argument('--data-path', type=str, required=True,
                    help='training and validation dataset.')
parser.add_argument('--batch-size', type=int, default=32,
                    help='training batch size per device (CPU/GPU).')
parser.add_argument('--dtype', type=str, default='float32',
                    help='data type for training. default is float32')
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
parser.add_argument('--input-size', type=int, default=224,
                    help='size of the input image size. default is 224')
parser.add_argument('--crop-ratio', type=float, default=0.875,
                    help='Crop ratio during validation. default is 0.875')
parser.add_argument('--mixup', action='store_true',
                    help='whether train the model with mix-up. default is false.')
parser.add_argument('--mixup-alpha', type=float, default=0.2,
                    help='beta distribution parameter for mixup sampling, default is 0.2.')
parser.add_argument('--label-smoothing', action='store_true',
                    help='use label smoothing or not in training. default is false.')
parser.add_argument('--no-wd', action='store_true',
                    help='whether to remove weight decay on bias, and beta/gamma for batchnorm layers.')
parser.add_argument('--last-gamma', action='store_true',
                    help='apply zero last bn weight in Bottleneck')
parser.add_argument('--sgd-gc', action='store_true',
                    help='using sgd Gradient Centralization')
parser.add_argument('--autoaugment', action='store_true',
                    help='use autoaugment')
parser.add_argument('--drop-block', action='store_true',
                    help='use DropBlock')
parser.add_argument('--save-dir', type=str, default='params',
                    help='directory of saved models')
parser.add_argument('--model-info', action='store_true',
                    help='show model information.')
parser.add_argument('--log-interval', type=int, default=50,
                    help='Number of batches to wait before logging.')
parser.add_argument('--logging-file', type=str, default='distribute_train_imagenet.log',
                    help='name of training log file')
parser.add_argument('--resume-epoch', type=int, default=0,
                    help='epoch to resume training from.')
parser.add_argument('--resume-param', type=str, default='',
                    help='resume training param path.')
parser.add_argument('--dist-url', default='tcp://127.0.0.1:26548', type=str,
                    help='url used to set up distributed training')
parser.add_argument("--rank", required=True, type=int,
                    help='node rank for distributed training')
parser.add_argument('--world-size', required=True, type=int,
                    help='number of nodes for distributed training')
# Default enable
# parser.add_argument('--multiprocessing-distributed', action='store_true',
#                     help='Use multi-processing distributed training to launch '
#                          'N processes per node, which has N GPUs. This is the '
#                          'fastest way to use PyTorch for either single node or '
#                          'multi node data parallel training')

warnings.filterwarnings("ignore", "(Possibly )?corrupt EXIF data", UserWarning)
assert torch.cuda.is_available(), \
    "Please don't waste of your time,it's impossible to train on CPU."


class ZeroLastGamma(object):
    def __init__(self, block_name='Bottleneck', bn_name='bn3'):
        self.block_name = block_name
        self.bn_name = bn_name

    def __call__(self, module):
        if module.__class__.__name__ == self.block_name:
            target_bn = module.__getattr__(self.bn_name)
            nn.init.zeros_(target_bn.weight)


def main():
    args = parser.parse_args()
    logger = get_logger(args.logging_file)
    logger.info(args)
    args.save_dir = os.path.join(os.getcwd(), args.save_dir)
    check_dir(args.save_dir)

    assert args.world_size >= 1

    args.classes = 1000
    args.num_training_samples = 1281167
    args.world = args.rank
    ngpus_per_node = torch.cuda.device_count()
    args.world_size = ngpus_per_node * args.world_size
    args.mix_precision_training = True if args.dtype == 'float16' else False
    mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))


def main_worker(gpu, ngpus_per_node, args):
    args.gpu = gpu
    logger = get_logger(args.logging_file)
    logger.info("Use GPU: {} for training".format(args.gpu))

    args.rank = args.rank * ngpus_per_node + gpu
    torch.distributed.init_process_group(backend="nccl", init_method=args.dist_url,
                                         world_size=args.world_size, rank=args.rank)

    epochs = args.epochs
    input_size = args.input_size
    resume_epoch = args.resume_epoch
    initializer = KaimingInitializer()
    zero_gamma = ZeroLastGamma()
    is_first_rank = True if args.rank % ngpus_per_node == 0 else False

    batches_pre_epoch = args.num_training_samples // (args.batch_size * ngpus_per_node)
    lr = 0.1 * (args.batch_size * ngpus_per_node // 32) if args.lr == 0 else args.lr

    model = get_model(models, args.model)

    model.apply(initializer)
    if args.last_gamma:
        model.apply(zero_gamma)
        logger.info('Apply zero last gamma init.')

    if is_first_rank and args.model_info:
        summary(model, torch.rand((1, 3, input_size, input_size)))

    parameters = model.parameters() if not args.no_wd else no_decay_bias(model)
    if args.sgd_gc:
        logger.info('Use SGD_GC optimizer.')
        optimizer = SGD_GC(parameters, lr=lr, momentum=args.momentum,
                           weight_decay=args.wd, nesterov=True)
    else:
        optimizer = optim.SGD(parameters, lr=lr, momentum=args.momentum,
                              weight_decay=args.wd, nesterov=True)

    lr_scheduler = CosineWarmupLr(optimizer, batches_pre_epoch, epochs,
                                  base_lr=args.lr, warmup_epochs=args.warmup_epochs)

    # dropblock_scheduler = DropBlockScheduler(model, batches_pre_epoch, epochs)

    if args.lookahead:
        optimizer = Lookahead(optimizer)
        logger.info('Use lookahead optimizer.')

    torch.cuda.set_device(args.gpu)
    model.cuda(args.gpu)
    args.num_workers = int((args.num_workers + ngpus_per_node - 1) / ngpus_per_node)

    if args.mix_precision_training and is_first_rank:
        logger.info('Train with FP16.')

    scaler = GradScaler(enabled=args.mix_precision_training)
    model = nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])

    Loss = nn.CrossEntropyLoss().cuda(args.gpu) if not args.label_smoothing else \
        LabelSmoothingLoss(args.classes, smoothing=0.1).cuda(args.gpu)

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    if args.autoaugment:
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(input_size),
            transforms.RandomHorizontalFlip(),
            ImageNetPolicy,
            transforms.ToTensor(),
            normalize,
        ])
    else:
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(input_size),
            # Cutout(),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.4, 0.4, 0.4),
            transforms.ToTensor(),
            normalize,
        ])

    val_transform = transforms.Compose([
        transforms.Resize(int(input_size / 0.875)),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        normalize,
    ])

    train_set = ImageNet(args.data_path, split='train', transform=train_transform)
    val_set = ImageNet(args.data_path, split='val', transform=val_transform)

    train_sampler = DistributedSampler(train_set)
    train_loader = DataLoader(train_set, args.batch_size, False, pin_memory=True,
                              num_workers=args.num_workers, drop_last=True, sampler=train_sampler)
    val_loader = DataLoader(val_set, args.batch_size, False, pin_memory=True, num_workers=args.num_workers,
                            drop_last=False)

    if resume_epoch > 0:
        loc = 'cuda:{}'.format(args.gpu)
        checkpoint = torch.load(args.resume_param, map_location=loc)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scaler.load_state_dict(checkpoint['scaler'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        print("Finish loading resume param.")

    torch.backends.cudnn.benchmark = True

    top1_acc = metric.Accuracy(name='Top1 Accuracy')
    top5_acc = metric.TopKAccuracy(top=5, name='Top5 Accuracy')
    loss_record = metric.NumericalCost(name='Loss')

    for epoch in range(resume_epoch, epochs):
        tic = time.time()
        train_sampler.set_epoch(epoch)
        if not args.mixup:
            train_one_epoch(model, train_loader, Loss, optimizer, epoch, lr_scheduler,
                            logger, top1_acc, loss_record, scaler, args)
        else:
            train_one_epoch_mixup(model, train_loader, Loss, optimizer, epoch, lr_scheduler,
                                  logger, loss_record, scaler, args)
        train_speed = int(args.num_training_samples // (time.time() - tic))
        if is_first_rank:
            logger.info('Finish one epoch speed: {} samples/s'.format(train_speed))
        test(model, val_loader, Loss, epoch, logger, top1_acc, top5_acc, loss_record, args)

        if args.rank % ngpus_per_node == 0:
            checkpoint = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scaler': scaler.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
            }
            torch.save(checkpoint, '{}/{}_{}_{:.5}.pt'.format(
                args.save_dir, args.model, epoch, top1_acc.get()))


@torch.no_grad()
def test(model, val_loader, criterion, epoch, logger, top1_acc, top5_acc, loss_record, args):
    top1_acc.reset()
    top5_acc.reset()
    loss_record.reset()

    model.eval()
    for data, labels in val_loader:
        data = data.cuda(args.gpu, non_blocking=True)
        labels = labels.cuda(args.gpu, non_blocking=True)

        outputs = model(data)
        losses = criterion(outputs, labels)

        top1_acc.update(outputs, labels)
        top5_acc.update(outputs, labels)
        loss_record.update(losses)

    test_msg = 'Test Epoch {}, Node {}, GPU {}: {}:{:.5}, {}:{:.5}, {}:{:.5}'.format(
        epoch, args.world, args.gpu, top1_acc.name, top1_acc.get(), top5_acc.name,
        top5_acc.get(), loss_record.name, loss_record.get())
    logger.info(test_msg)


def train_one_epoch(model, train_loader, criterion, optimizer, epoch, lr_scheduler,
                    logger, top1_acc, loss_record, scaler, args):
    top1_acc.reset()
    loss_record.reset()
    tic = time.time()

    model.train()
    for i, (data, labels) in enumerate(train_loader):
        data = data.cuda(args.gpu, non_blocking=True)
        labels = labels.cuda(args.gpu, non_blocking=True)

        optimizer.zero_grad()
        with autocast(enabled=args.mix_precision_training):
            outputs = model(data)
            loss = criterion(outputs, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        lr_scheduler.step()

        top1_acc.update(outputs, labels)
        loss_record.update(loss)

        if i % args.log_interval == 0 and i != 0:
            logger.info('Epoch {}, Node {}, GPU {}, Iter {}, {}:{:.5}, {}:{:.5}, {} samples/s. lr: {:.5}.'.format(
                epoch, args.world, args.gpu, i, top1_acc.name, top1_acc.get(),
                loss_record.name, loss_record.get(),
                int((i * args.batch_size) // (time.time() - tic)),
                lr_scheduler.learning_rate
            ))


def train_one_epoch_mixup(model, train_loader, criterion, optimizer, epoch, lr_scheduler,
                          logger, loss_record, scaler, args):
    loss_record.reset()
    tic = time.time()

    model.train()
    for i, (data, labels) in enumerate(train_loader):
        data = data.cuda(args.gpu, non_blocking=True)
        labels = labels.cuda(args.gpu, non_blocking=True)

        data, labels_a, labels_b, lam = mixup_data(data, labels, args.mixup_alpha)
        optimizer.zero_grad()
        with autocast(args.mix_precision_training):
            outputs = model(data)
            loss = mixup_criterion(criterion, outputs, labels_a, labels_b, lam)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        loss_record.update(loss)
        lr_scheduler.step()

        if i % args.log_interval == 0 and i != 0:
            logger.info('Epoch {}, Node {}, GPU {}, Iter {}, {}:{:.5}, {} samples/s, lr: {:.5}.'.format(
                epoch, args.world, args.gpu, i, loss_record.name, loss_record.get(),
                int((i * args.batch_size) // (time.time() - tic)),
                lr_scheduler.learning_rate))


if __name__ == '__main__':
    main()
