'''
Functions for data loading, cleaning, transforming
'''
import os

import torch
from torch.utils.data import random_split
import torchvision

import src.utils.config as cfg


def get_data(compute_stats=False):
    '''
    Get a given supported dataset and return both clean and perturbed data
    samples
    '''

    if cfg.DATASET == 'mnist':
        getter = torchvision.datasets.MNIST
    elif cfg.DATASET == 'emnist_balanced':
        getter = lambda *args, **kwargs: torchvision.datasets.EMNIST(
            *args, split='balanced', **kwargs
        )
    elif cfg.DATASET == 'cifar10':
        getter = torchvision.datasets.CIFAR10
    elif cfg.DATASET == 'cifar100':
        getter = torchvision.datasets.CIFAR100
    elif cfg.DATASET == 'imagenet':
        getter = torchvision.datasets.ImageNet
    else:
        raise NotImplementedError(
            f'Dataset {cfg.DATASET} not supported, \
            must be in {[*cfg.DATASET_CONFIGS.keys()]}')

    # Load datasets
    def transform(x, perturb=False):
        '''
        Transforms and manipulations to apply to images
        '''
        x = torchvision.transforms.ToTensor()(x)  # Pixels to range [0, 1]
        x = torchvision.transforms.RandomCrop(size=(cfg.OUT_SHAPE[1:]))(x)
        if perturb:
            x = torchvision.transforms.RandomAffine(
                degrees=0, translate=(cfg.HOR_SHIFT, cfg.VER_SHIFT)
            )(x)

        return x

    def dataset_factory(train=True, perturb=True):
        return getter(
            os.path.join('data'),
            transform=lambda x: transform(x, perturb=perturb), download=True,
            train=train
        )

    train_dataset = dataset_factory(train=True, perturb=False)
    test_dataset = dataset_factory(train=False, perturb=False)
    train_dataset_pert = dataset_factory(train=True, perturb=True)
    test_dataset_pert = dataset_factory(train=False, perturb=True)

    # Train val split:
    train_dataset, val_dataset = random_split(train_dataset, [4/5, 1/5])
    train_dataset_pert, val_dataset_pert = random_split(
        train_dataset_pert, [4/5, 1/5])

    # Compute and store stats
    if compute_stats:
        mean, std = comp_stats(train_dataset)
        cfg.MEAN = mean
        cfg.STD = std

    # Create PyTorch DataLoaders
    def dataloader_factory(dataset):
        return torch.utils.data.DataLoader(
            dataset, batch_size=cfg.BATCH_SIZE, shuffle=True,
            num_workers=cfg.NUM_WORKERS,
        )

    train_loader = dataloader_factory(train_dataset)
    val_loader = dataloader_factory(val_dataset)
    test_loader = dataloader_factory(test_dataset)

    train_loader_pert = dataloader_factory(train_dataset_pert)
    val_loader_pert = dataloader_factory(val_dataset_pert)
    test_loader_pert = dataloader_factory(test_dataset_pert)

    # Return loaders for train, val, test, for clean, pert data
    return train_loader, val_loader, test_loader, train_loader_pert, \
        val_loader_pert, test_loader_pert


def comp_stats(dataset):
    '''
    Inefficient way of computing mean and std of dataset
    '''

    sum_ = torch.zeros_like(dataset[0][0])
    for x, _ in dataset:
        sum_ += x

    mean = sum_.mean(axis=[-1, -2], keepdim=True) / len(dataset)

    square_sum = torch.zeros_like(dataset[0][0])
    for x, _ in dataset:
        square_sum += (x - mean) ** 2

    var = square_sum.mean(axis=[-1, -2]) / len(dataset)

    mean = mean.squeeze()
    std = var ** 0.5

    return mean, std


def standardise(x):
    '''
    Takes tensor of shape [..., C, H, W] and standardises all pixel values
    to a mean of 0 and unit variance.
    Returns standardised tensor
    '''
    if cfg.MEAN is None or cfg.STD is None:
        raise ValueError(
            'Dataset mean and std must be computed before standardisation'
        )

    mean = cfg.MEAN.reshape((-1, 1, 1))
    std = cfg.STD.reshape((-1, 1, 1))

    x_s = (x - mean) / std

    return x_s


def unstandardise(x_s):
    '''
    Inverse operation of standardise
    '''
    if cfg.MEAN is None or cfg.STD is None:
        raise ValueError(
            'Dataset mean and std must be computed before unstandardisation'
        )

    mean = cfg.MEAN.reshape((-1, 1, 1))
    std = cfg.STD.reshape((-1, 1, 1))

    x = x_s * std + mean

    return x
