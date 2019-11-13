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
parser.add_argument('--name', type=str, required=True,
                    help='Save file name, need not to add `.lmdb`')
parser.add_argument('--train', action='store_true', help='train or val dataset.')
parser.add_argument('--download', action='store_true', help='download dataset.')
parser.add_argument('-j', dest='num_workers', type=int, default=0)
parser.add_argument('--write-frequency', type=int, default=5000)
parser.add_argument('--max-size', type=float, default=1.0,
                    help='Maximum size database, this is rate, default is 1T, final setting would be '
                         '1T * `this param`')

args = parser.parse_args()
split = 'train' if args.train else 'val'
check_dir(args.save_dir)
data_set = ImageNet(args.data_dir, split, args.download, loader=raw_reader)

if __name__ == '__main__':
    generate_lmdb_dataset(data_set, args.save_dir, args.name, args.num_workers,
                          args.max_size, args.write_frequency)
