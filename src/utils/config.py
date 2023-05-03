'''
Global config variables accesible by all experiments.
Can be set and overridden with cmd args.
'''
import torch


# General
DEBUG = False
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
NUM_WORKERS = 8
# Data perturbations to apply
HOR_SHIFT = 0.2
VER_SHIFT = 0.0

# Dataset specific
DATASET = ''
OUT_SHAPE = ()
DIM_Y = None
DIM_Z = None
DIM_M = None
MEAN = None
STD = None

# Supported datasets and configs
DATASET_CONFIGS = {
    'mnist': {
        'out_shape': (1, 28, 28),
        'dim_y': 10,
        'dim_z': 64,
        'dim_m': 32
    },
    'emnist_balanced': {
        'out_shape': (1, 28, 28),
        'dim_y': 47,
        'dim_z': 128,
        'dim_m': 64
    },
    'cifar10': {
        'out_shape': (3, 32, 32),
        'dim_y': 10,
        'dim_z': 128,
        'dim_m': 64
    },
    'cifar100': {
        'out_shape': (3, 32, 32),
        'dim_y': 100,
        'dim_z': 128,
        'dim_m': 64
    },
    'imagenet': {
        'out_shape': (3, 224, 224),
        'dim_y': 100,
        'dim_z': 256,
        'dim_m': 128,
    },
}

# Training
NUM_EPOCHS = 250
BATCH_SIZE = 128
BETA = 0.5              # Scale KL-divergence up or down
LAMBDA = 0.85           # Scale contribution of clean vs perturbed data

# Eval
