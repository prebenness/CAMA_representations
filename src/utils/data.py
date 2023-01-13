'''
Functions for data loading, cleaning, transforming
'''
import os

import torch
import torchvision

import src.utils.config as cfg


def get_data(compute_stats=False):
    '''
    Get a given supported dataset and return both clean and perturbed data
    samples
    '''

    def transform(x, perturb=False):
        '''
        Transforms and manipulations to apply to images
        '''
        x = torchvision.transforms.ToTensor()(x)  # Pixels to range [0, 1]
        if perturb:
            x = torchvision.transforms.RandomAffine(
                degrees=0, translate=(cfg.HOR_SHIFT, cfg.VER_SHIFT)
            )(x)

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

    # Load datasets
    train_dataset = getter(
        os.path.join('data'), transform=lambda x: transform(x, perturb=False),
        download=True, train=True
    )
    test_dataset = getter(
        os.path.join('data'), transform=lambda x: transform(x, perturb=False),
        download=True, train=False
    )

    train_dataset_pert = getter(
        os.path.join('data'), transform=lambda x: transform(x, perturb=True),
        download=True, train=True
    )
    test_dataset_pert = getter(
        os.path.join('data'), transform=lambda x: transform(x, perturb=True),
        download=True, train=False
    )

    # Compute and store stats
    if compute_stats:
        mean, std = comp_stats(train_dataset)
        cfg.MEAN = mean
        cfg.STD = std

    # Create PyTorch DataLoaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=cfg.BATCH_SIZE, shuffle=True,
        num_workers=cfg.NUM_WORKERS
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=cfg.BATCH_SIZE, shuffle=True,
        num_workers=cfg.NUM_WORKERS
    )

    train_loader_pert = torch.utils.data.DataLoader(
        train_dataset_pert, batch_size=cfg.BATCH_SIZE, shuffle=True,
        num_workers=cfg.NUM_WORKERS
    )
    test_loader_pert = torch.utils.data.DataLoader(
        test_dataset_pert, batch_size=cfg.BATCH_SIZE, shuffle=True,
        num_workers=cfg.NUM_WORKERS
    )

    return train_loader, test_loader, train_loader_pert, test_loader_pert


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
