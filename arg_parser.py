import argparse
from src import config

import os


def get_args_parser():
    parser = argparse.ArgumentParser('Retinaface Training', add_help=False)
    parser.add_argument('--training_dataset', default=config.WIDER_TRAIN_LABEL_TXT, help='Training dataset directory')
    parser.add_argument('--network', default='resnet50', help='Backbone network mobile0.25 or resnet50')
    parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float, help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
    parser.add_argument('--checkpoint', default=None, help='resume net for retraining')
    parser.add_argument('--start_epoch', default=1, type=int, help='resume iter for retraining')
    parser.add_argument('--weight_decay', default=5e-4, type=float, help='Weight decay for SGD')
    parser.add_argument('--gamma', default=0.1, type=float, help='Gamma update for SGD')
    parser.add_argument('--save_folder', default='./weights/', help='Location to save checkpoint models')
    parser.add_argument('--output_dir', default='./weights/', help='Location to save checkpoint models')

    return parser


def get_args():
    parser = argparse.ArgumentParser('RetinaFace Training Script', parents=[get_args_parser()])
    args = parser.parse_args()
    return args
