# -*- coding: utf-8 -*-
# @Author  : DevinYang(pistonyang@gmail.com)

import argparse, time, logging, os
import models
import torch
import warnings
import apex
from torch.utils.data import DistributedSampler

from torchtoolbox import metric
from torchtoolbox.transform import Cutout
from torchtoolbox.nn import LabelSmoothingLoss, SwitchNorm2d, Swish
from torchtoolbox.optimizer import CosineWarmupLr, Lookahead
from torchtoolbox.nn.init import KaimingInitializer
from torchtoolbox.tools import no_decay_bias, \
    mixup_data, mixup_criterion, check_dir
from torchtoolbox.data import ImageLMDB

from torchvision.datasets import ImageNet
from torch import multiprocessing as mp
from torch.utils.data import DataLoader
from torchvision import transforms
from torch import nn
from torch import optim
from apex import amp
from apex.parallel.distributed import DistributedDataParallel as DDP

parser = argparse.ArgumentParser(description='Train a model on ImageNet.')
parser.add_argument('--data-path', type=str, required=True,
                    help='training and validation dataset.')
parser.add_argument('--use-lmdb', action='store_true',
                    help='use LMDB dataset/format')
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
parser.add_argument('--input-size', type=int, default=224,
                    help='size of the input image size. default is 224')
parser.add_argument('--crop-ratio', type=float, default=0.875,
                    help='Crop ratio during validation. default is 0.875')
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
parser.add_argument('--logging-file', type=str, default='train_imagenet.log',
                    help='name of training log file')
parser.add_argument('--resume-epoch', type=int, default=0,
                    help='epoch to resume training from.')
parser.add_argument('--resume-param', type=str, default='',
                    help='resume training param path.')
parser.add_argument('--dist-url', default='tcp://127.0.0.1:FREEPORT', type=str,
                    help='url used to set up distributed training')
parser.add_argument("--rank", default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--world-size', default=-1, type=int,
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


def get_model(name, **kwargs):
    return models.__dict__[name](**kwargs)


def set_model(drop_out, norm_layer, act, args):
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


def main():
    args = parser.parse_args()

    filehandler = logging.FileHandler(args.logging_file)
    streamhandler = logging.StreamHandler()

    logger = logging.getLogger('')
    logger.setLevel(logging.INFO)
    logger.addHandler(filehandler)
    logger.addHandler(streamhandler)
    logger.info(args)

    check_dir(args.save_dir)

    assert args.world_size >= 1

    args.classes = 1000
    args.num_training_samples = 1281167
    ngpus_per_node = torch.cuda.device_count()
    args.world_size = ngpus_per_node * args.world_size
    mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))


def main_worker(gpu, ngpus_per_node, args):
    args.gpu = gpu
    print("Use GPU: {} for training".format(args.gpu))

    args.rank = args.rank * ngpus_per_node + gpu
    torch.distributed.init_process_group(backend="nccl", init_method=args.dist_url,
                                         world_size=args.world_size, rank=args.rank)

    dtype = args.dtype
    epochs = args.epochs
    resume_epoch = args.resume_epoch
    batches_pre_epoch = args.num_training_samples // (args.batch_size * ngpus_per_node)
    lr = 0.1 * (args.batch_size * ngpus_per_node // 32) if args.lr == 0 else args.lr

    model_setting = set_model(args.dropout, args.norm_layer, args.activation, args)

    try:
        model = get_model(args.model, alpha=args.alpha, **model_setting)
    except TypeError:
        model = get_model(args.model, **model_setting)

    KaimingInitializer(model)

    if args.sync_bn:
        print('Use Apex Synced BN.')
        model = apex.parallel.convert_syncbn_model(model)

    parameters = model.parameters() if not args.no_wd else no_decay_bias(model)
    optimizer = optim.SGD(parameters, lr=lr, momentum=args.momentum,
                          weight_decay=args.wd, nesterov=True)
    lr_scheduler = CosineWarmupLr(optimizer, batches_pre_epoch, epochs,
                                  base_lr=args.lr, warmup_epochs=args.warmup_epochs)
    if args.lookahead:
        print('Use lookahead optimizer.')
        optimizer = Lookahead(optimizer)

    torch.cuda.set_device(args.gpu)
    model.cuda(args.gpu)
    args.num_workers = int((args.num_workers + ngpus_per_node - 1) / ngpus_per_node)

    if dtype == 'float16':
        print('Train with FP16.')
        model, optimizer = amp.initialize(model, optimizer, opt_level='O1')
    model = nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
    # model = DDP(model, delay_allreduce=True)

    Loss = nn.CrossEntropyLoss().cuda(args.gpu) if not args.label_smoothing else \
        LabelSmoothingLoss(args.classes, smoothing=0.1).cuda(args.gpu)

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        # Cutout(),
        # transforms.RandomRotation(15),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.4, 0.4, 0.4),
        transforms.ToTensor(),
        normalize,
    ])

    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])

    if not args.use_lmdb:
        train_set = ImageNet(args.data_path, split='train', transform=train_transform)
        val_set = ImageNet(args.data_path, split='val', transform=val_transform)
    else:
        train_set = ImageLMDB(os.path.join(args.data_path, 'train.lmdb'), transform=train_transform)
        val_set = ImageLMDB(os.path.join(args.data_path, 'val.lmdb'), transform=val_transform)

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
        amp.load_state_dict(checkpoint['amp'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        print("Finish loading resume param.")

    torch.backends.cudnn.benchmark = True

    for epoch in range(resume_epoch, epochs):
        tic = time.time()
        train_sampler.set_epoch(epoch)
        if not args.mixup:
            train_one_epoch(model, train_loader, Loss, optimizer, epoch, lr_scheduler, args)
        else:
            train_one_epoch_mixup(model, train_loader, Loss, optimizer, epoch, lr_scheduler, args)
        train_speed = int(args.num_training_samples // (time.time() - tic))
        print('Finish one epoch speed: {} samples/s'.format(train_speed))
        top1_acc = test(model, val_loader, Loss, epoch, args)

        if args.rank % ngpus_per_node == 0:
            checkpoint = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'amp': amp.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
            }
            torch.save(checkpoint, '{}/{}_{}_{:.5}.pt'.format(
                args.save_dir, args.model, epoch, top1_acc))


@torch.no_grad()
def test(model, val_loader, criterion, epoch, args):
    top1_acc = metric.Accuracy(name='Top1 Accuracy')
    top5_acc = metric.TopKAccuracy(top=5, name='Top5 Accuracy')
    loss_record = metric.NumericalCost(name='Loss')
    model.eval()
    for data, labels in val_loader:
        data = data.cuda(args.gpu, non_blocking=True)
        labels = labels.cuda(args.gpu, non_blocking=True)

        outputs = model(data)
        losses = criterion(outputs, labels)

        top1_acc.step(outputs, labels)
        top5_acc.step(outputs, labels)
        loss_record.step(losses)

    test_msg = 'Test Epoch {}: {}:{:.5}, {}:{:.5}, {}:{:.5}\n'.format(
        epoch, top1_acc.name, top1_acc.get(), top5_acc.name, top5_acc.get(),
        loss_record.name, loss_record.get())
    print(test_msg)
    return top1_acc.get()


def train_one_epoch(model, train_loader, criterion, optimizer, epoch, lr_scheduler, args):
    top1_acc = metric.Accuracy(name='Top1 Accuracy')
    loss_record = metric.NumericalCost(name='Loss')
    tic = time.time()

    model.train()
    for i, (data, labels) in enumerate(train_loader):
        data = data.cuda(args.gpu, non_blocking=True)
        labels = labels.cuda(args.gpu, non_blocking=True)

        optimizer.zero_grad()
        outputs = model(data)
        loss = criterion(outputs, labels)

        with amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward()
        optimizer.step()

        lr_scheduler.step()
        top1_acc.step(outputs, labels)
        loss_record.step(loss)

        if i % args.log_interval == 0 and i != 0:
            print('Epoch {}, Iter {}, {}:{:.5}, {}:{:.5}, {} samples/s. lr: {:.5}.'.format(
                epoch, i, top1_acc.name, top1_acc.get(),
                loss_record.name, loss_record.get(),
                int((i * args.batch_size) // (time.time() - tic)),
                lr_scheduler.learning_rate
            ))


def train_one_epoch_mixup(model, train_loader, criterion, optimizer, epoch, lr_scheduler, args):
    loss_record = metric.NumericalCost(name='Loss')
    mixup_off_epoch = args.epochs if args.mixup_off_epoch == 0 else args.mixup_off_epoch

    alpha = args.mixup_alpha if epoch < mixup_off_epoch else 0
    tic = time.time()

    model.train()
    for i, (data, labels) in enumerate(train_loader):
        data = data.cuda(args.gpu, non_blocking=True)
        labels = labels.cuda(args.gpu, non_blocking=True)

        data, labels_a, labels_b, lam = mixup_data(data, labels, alpha)
        optimizer.zero_grad()
        outputs = model(data)
        loss = mixup_criterion(criterion, outputs, labels_a, labels_b, lam)

        with amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward()
        optimizer.step()

        loss_record.step(loss)
        lr_scheduler.step()

        if i % args.log_interval == 0 and i != 0:
            print('Epoch {}, Iter {}, {}:{:.5}, {} samples/s.'.format(
                epoch, i, loss_record.name, loss_record.get(),
                int((i * args.batch_size) // (time.time() - tic))
            ))


if __name__ == '__main__':
    main()
