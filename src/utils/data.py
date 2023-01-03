'''
Functions for data loading, cleaning, transforming
'''
import os

import torch
import torchvision

import src.utils.config as cfg


def get_data():
    '''
    Get a given supported dataset and return both clean and perturbed data
    samples
    '''

    # TODO: standardise pixel values, p(x) = N(0,1)
    def transform(x):
        # Always normalise pixels to range [0.0, 1.0]
        x = torchvision.transforms.ToTensor()(x)
        x = torchvision.transforms.RandomAffine(
            degrees=0, translate=(cfg.HOR_SHIFT, cfg.VER_SHIFT))(x)
        return x

    if cfg.DATASET == 'mnist':
        getter = torchvision.datasets.MNIST
    elif cfg.DATASET == 'cifar10':
        getter = torchvision.datasets.CIFAR10
    elif cfg.DATASET == 'cifar100':
        getter = torchvision.datasets.CIFAR100
    else:
        raise NotImplementedError(
            f'Dataset {cfg.DATASET} not supported, \
            must be either "mnist", "cifar10", or "cifar100"')

    dataset = getter(os.path.join('data'), transform=transform, download=True)
    data = torch.utils.data.DataLoader(
        dataset, batch_size=cfg.BATCH_SIZE, shuffle=True,
        num_workers=cfg.NUM_WORKERS
    )

    return data
