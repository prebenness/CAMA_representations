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
HOR_SHIFT = 0.5
VER_SHIFT = 0.0

# Dataset specific
DATASET = 'mnist'
OUT_SHAPE = (1, 28, 28)
DIM_Y = 10
DIM_Z = 64
DIM_M = 32
MEAN = None
STD = None

# Training
NUM_EPOCHS = 250
BATCH_SIZE = 128
BETA = 1.0
LAMBDA = 0.7

# Eval
