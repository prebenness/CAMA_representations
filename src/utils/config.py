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

# Training
NUM_EPOCHS = 250
BATCH_SIZE = 128
BETA = 0.5              # Scale KL-divergence up or down
LAMBDA = 0.85           # Scale contribution of clean vs perturbed data

# Eval
