'''
Parse cmd line args, type check and override relevant default config values
'''
import argparse

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


def update_config(args):
    '''
    Lets cmd line args override default values for global config variables
    '''
    # Set dataset variables
    if args.dataset == 'mnist':
        cfg.DATASET = 'mnist'
        cfg.DIM_Y = 10
        cfg.DIM_Z = 64
        cfg.DIM_M = 32
        cfg.OUT_SHAPE = (1, 28, 28)
    elif args.dataset == 'cifar10':
        cfg.DATASET = 'cifar10'
        cfg.DIM_Y = 10
        cfg.DIM_Z = 64
        cfg.DIM_M = 32
        cfg.OUT_SHAPE = (3, 32, 32)
    elif args.dataset == 'cifar100':
        cfg.DIM_Y = 100
        cfg.DIM_Z = 64
        cfg.DIM_M = 32
        cfg.OUT_SHAPE = (3, 32, 32)

    # Store rest
    cfg.DEBUG = args.debug
    cfg.BETA = args.beta
    cfg.LAMBDA = args.lambda_

    if args.debug:
        cfg.DEBUG = True


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

    # Train args
    parser.add_argument('-b', '--beta', type=pos_float)
    parser.add_argument('-l', '--lambda', dest='lambda_', type=pos_float)
    parser.add_argument('--num_epochs', type=int)
    parser.add_argument('--batch_size', type=int)

    # Eval args

    # Representation args

    # Parse args and update global config variables
    args = parser.parse_args()
    update_config(args)
    return args
