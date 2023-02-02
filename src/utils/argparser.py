'''
Parse cmd line args, type check and override relevant default config values
'''
import argparse
import os
from datetime import datetime

import src.utils.config as cfg


def percentage(f):
    '''
    Return input iff float between 0.0 and 1.0 inclusive
    '''
    f = float(f)
    if f > 1.0 or f < 0.0:
        raise ValueError(
            'Value must be float in the closed interval [0.0, 1.0]')

    return f


def pos_float(f):
    '''
    Return input iff positive float
    '''
    f = float(f)
    if f < 0.0:
        raise ValueError('Value must be positive float')

    return f


def file_path(s):
    if os.path.isfile(s):
        return s
    else:
        raise FileNotFoundError(f'File {s} not found')


def update_config(args):
    '''
    Lets cmd line args override default values for global config variables
    '''
    # Set dataset variables
    cfg.DATASET = args.dataset

    if args.dataset == 'mnist':
        cfg.DATASET = 'mnist'
        cfg.DIM_Y = 10
        cfg.DIM_Z = 64
        cfg.DIM_M = 32
        cfg.OUT_SHAPE = (1, 28, 28)
    elif args.dataset == 'cifar10':
        cfg.DATASET = 'cifar10'
        cfg.DIM_Y = 10
        cfg.DIM_Z = 128
        cfg.DIM_M = 64
        cfg.OUT_SHAPE = (3, 32, 32)
    elif args.dataset == 'cifar100':
        cfg.DIM_Y = 100
        cfg.DIM_Z = 128
        cfg.DIM_M = 64
        cfg.OUT_SHAPE = (3, 32, 32)

    # Overwrite defaults if cmd args given
    cfg.HOR_SHIFT = args.hor_shift or cfg.HOR_SHIFT
    cfg.VER_SHIFT = args.ver_shift or cfg.VER_SHIFT
    cfg.DEBUG = args.debug or cfg.DEBUG

    # Train args
    cfg.BETA = args.beta or cfg.BETA
    cfg.LAMBDA = args.lambda_ or cfg.LAMBDA
    cfg.NUM_EPOCHS = args.num_epochs or cfg.NUM_EPOCHS
    cfg.BATCH_SIZE = args.batch_size or cfg.BATCH_SIZE

    # Make results dir if needed
    if args.trained_model:
        out_dir = os.path.join(args.trained_model.split(os.path.sep)[:-2])
    else:
        name_list = [
            args.exp_name, f'epochs={cfg.NUM_EPOCHS}', f'{cfg.DATASET}',
            datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        ]

        if cfg.DEBUG:
            name_list = ['DEBUG'] + name_list

        name = '_'.join(name_list)

        out_dir = os.path.join('results', name)
        os.makedirs(out_dir, exist_ok=True)

    cfg.OUT_DIR = out_dir


def parse_args():
    '''
    Parse and sanitise command line arguments
    '''

    parser = argparse.ArgumentParser(
        prog='DeepCAMA', description='Train, \
            test, or compute representations from \
            a DeepCAMA model'
    )

    # General args
    parser.add_argument(
        '-m', '--mode', choices=['train', 'finetune', 'test', 'repr'],
        required=True
    )
    parser.add_argument(
        '-d', '--dataset', choices=['mnist', 'cifar10', 'cifar100'],
        required=True
    )
    parser.add_argument('-hor', '--hor_shift', type=percentage)
    parser.add_argument('-ver', '--ver_shift', type=percentage)
    parser.add_argument('--debug', action='store_true', default=False)
    parser.add_argument('--exp_name', '-n', type=str, default='test')

    # Train args
    parser.add_argument('-b', '--beta', type=pos_float)
    parser.add_argument('-l', '--lambda', dest='lambda_', type=pos_float)
    parser.add_argument('--num_epochs', type=int)
    parser.add_argument('--batch_size', type=int)

    # Eval args
    parser.add_argument('--trained_model', type=file_path)

    # Representation args

    # Parse args and update global config variables
    args = parser.parse_args()
    update_config(args)
    return args
