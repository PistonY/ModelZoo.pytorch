# -*- coding: utf-8 -*-
# @Author  : DevinYang(pistonyang@gmail.com)
import argparse
from torchvision.datasets import ImageNet
from torchtoolbox.tools import check_dir
from torchtoolbox.tools.convert_lmdb import generate_lmdb_dataset, raw_reader

parser = argparse.ArgumentParser(description='Convert a ImageFolder dataset to LMDB format.')
parser.add_argument('--data-dir', type=str, required=True,
                    help='ImageFolder path, this param will give to ImageFolder Dataset.')
parser.add_argument('--save-dir', type=str, required=True,
                    help='Save dir.')
parser.add_argument('--download', action='store_true', help='download dataset.')
parser.add_argument('-j', dest='num_workers', type=int, default=0)
parser.add_argument('--write-frequency', type=int, default=5000)
parser.add_argument('--max-size', type=float, default=1.0,
                    help='Maximum size database, this is rate, default is 1T, final setting would be '
                         '1T * `this param`')

args = parser.parse_args()
check_dir(args.save_dir)
train_data_set = ImageNet(args.data_dir, 'train', args.download, loader=raw_reader)
val_data_set = ImageNet(args.data_dir, 'val', args.download, loader=raw_reader)

if __name__ == '__main__':
    generate_lmdb_dataset(train_data_set, args.save_dir, 'train', args.num_workers,
                          args.max_size, args.write_frequency)
    generate_lmdb_dataset(val_data_set, args.save_dir, 'val', args.num_workers,
                          args.max_size, args.write_frequency)
