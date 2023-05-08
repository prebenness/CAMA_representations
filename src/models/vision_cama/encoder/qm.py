'''
Variational encoder which estimates mean and std of q(m|x) and can return one
sample using the reparameterization trick
'''
import torch
from torch import nn
import torch.nn.functional as F

import src.utils.config as cfg
from src.models.vision_cama.encoder.input_encoders.cnns import\
    CNNMNIST, CNNCIFAR, CNNIMAGENET, CNNMedium
from src.models.vision_cama.encoder.input_encoders.resnet import ResNetEmbedder


class QMMNIST(nn.Module):
    ''' Parameterises q(m|x)
    '''

    def __init__(self, dim_m):
        super(QMMNIST, self).__init__()

        self.dim_m = dim_m

        # Input encoding
        self.input_encoder = CNNMNIST(kernel_size=3)

        # MLP layers
        # [ 4*4*64, 500, dim_m*2 ]
        self.linear1 = nn.Linear(in_features=4*4*64, out_features=500)
        self.linear_mu = nn.Linear(in_features=500, out_features=dim_m)
        self.linear_logvar = nn.Linear(in_features=500, out_features=dim_m)

        # Distribution for reparam trick
        self.std_normal = torch.distributions.Normal(0, 1)
        if cfg.DEVICE == 'cuda':
            self.std_normal.loc = self.std_normal.loc.cuda()
            self.std_normal.scale = self.std_normal.scale.cuda()

        # Track KL divergence
        self.kl = 0

    def forward(self, x):
        # Encode input
        x = self.input_encoder(x)

        # Reshape to vector
        x = torch.flatten(x, start_dim=1)

        # MLP layer
        x = F.relu(self.linear1(x))

        # Generate mean and logvar
        mu = self.linear_mu(x)
        sigma = torch.exp(0.5 * self.linear_logvar(x))

        # Reparameterization trick
        m = mu + sigma * self.std_normal.sample(mu.shape)

        self.kl = (
            (sigma ** 2) / 2 +
            (mu ** 2) / 2 -
            torch.log(sigma) -
            1 / 2
        ).sum()

        return m


class QMCIFAR(nn.Module):
    ''' Parameterises q(m|x)
    '''

    def __init__(self, dim_m):
        super(QMCIFAR, self).__init__()

        self.dim_m = dim_m

        # Input encoding
        self.input_encoder = CNNCIFAR()

        # MLP layers
        # [ 4*4*64, 1000, 1000, dim_m*2 ]
        self.linear1 = nn.Linear(in_features=4*4*64, out_features=1000)
        self.linear2 = nn.Linear(in_features=1000, out_features=1000)
        self.linear_mu = nn.Linear(in_features=1000, out_features=dim_m)
        self.linear_logvar = nn.Linear(in_features=1000, out_features=dim_m)

        # Distribution for reparam trick
        self.std_normal = torch.distributions.Normal(0, 1)
        if cfg.DEVICE == 'cuda':
            self.std_normal.loc = self.std_normal.loc.cuda()
            self.std_normal.scale = self.std_normal.scale.cuda()

        # Track KL divergence
        self.kl = 0

    def forward(self, x):
        # Encode input
        x = self.input_encoder(x)

        # Reshape to vector
        x = torch.flatten(x, start_dim=1)

        # MLP layers
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))

        # Generate mean and logvar
        mu = self.linear_mu(x)
        sigma = torch.exp(0.5 * self.linear_logvar(x))

        # Reparameterization trick
        m = mu + sigma * self.std_normal.sample(mu.shape)

        self.kl = (
            (sigma ** 2) / 2 +
            (mu ** 2) / 2 -
            torch.log(sigma) -
            1 / 2
        ).sum()

        return m


class QMIMAGENET(nn.Module):
    ''' Parameterises q(m|x)
    '''

    def __init__(self, dim_m):
        super(QMIMAGENET, self).__init__()

        self.dim_m = dim_m

        # Input encoding
        self.input_encoder = CNNIMAGENET()

        # MLP layers
        # [ 8*8*64, 1000, 1000, dim_m*2 ]
        self.linear1 = nn.Linear(in_features=8*8*64, out_features=1000)
        self.linear2 = nn.Linear(in_features=1000, out_features=1000)
        self.linear_mu = nn.Linear(in_features=1000, out_features=dim_m)
        self.linear_logvar = nn.Linear(in_features=1000, out_features=dim_m)

        # Distribution for reparam trick
        self.std_normal = torch.distributions.Normal(0, 1)
        if cfg.DEVICE == 'cuda':
            self.std_normal.loc = self.std_normal.loc.cuda()
            self.std_normal.scale = self.std_normal.scale.cuda()

        # Track KL divergence
        self.kl = 0

    def forward(self, x):
        # Encode input
        x = self.input_encoder(x)

        # Reshape to vector
        x = torch.flatten(x, start_dim=1)

        # MLP layers
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))

        # Generate mean and logvar
        mu = self.linear_mu(x)
        sigma = torch.exp(0.5 * self.linear_logvar(x))

        # Reparameterization trick
        m = mu + sigma * self.std_normal.sample(mu.shape)

        self.kl = (
            (sigma ** 2) / 2 +
            (mu ** 2) / 2 -
            torch.log(sigma) -
            1 / 2
        ).sum()

        return m


class QMRESNET(nn.Module):
    ''' Parameterises q(m|x)
    '''

    def __init__(self, dim_m):
        super(QMRESNET, self).__init__()

        self.dim_m = dim_m

        # Input encoding
        self.input_encoder = ResNetEmbedder(
            resnet_type=18, num_channels=cfg.OUT_SHAPE[0])

        # MLP layers
        self.linear_mu = nn.Linear(in_features=512, out_features=dim_m)
        self.linear_logvar = nn.Linear(in_features=512, out_features=dim_m)

        # Distribution for reparam trick
        self.std_normal = torch.distributions.Normal(0, 1)
        if cfg.DEVICE == 'cuda':
            self.std_normal.loc = self.std_normal.loc.cuda()
            self.std_normal.scale = self.std_normal.scale.cuda()

        # Track KL divergence
        self.kl = 0

    def forward(self, x):
        # Encode input
        x = self.input_encoder(x)

        # Generate mean and logvar
        mu = self.linear_mu(x)
        sigma = torch.exp(0.5 * self.linear_logvar(x))

        # Reparameterization trick
        m = mu + sigma * self.std_normal.sample(mu.shape)

        self.kl = (
            (sigma ** 2) / 2 +
            (mu ** 2) / 2 -
            torch.log(sigma) -
            1 / 2
        ).sum()

        return m


class QMMedium(nn.Module):
    ''' Parameterises q(m|x)
    '''

    def __init__(self, dim_m):
        super(QMMedium, self).__init__()

        self.dim_m = dim_m

        # Input encoding
        self.input_encoder = CNNMedium()

        # MLP layers
        # [ 8*8*64, 1000, 1000, dim_m*2 ]
        self.linear1 = nn.Linear(in_features=4*4*64, out_features=1000)
        self.linear2 = nn.Linear(in_features=1000, out_features=1000)
        self.linear_mu = nn.Linear(in_features=1000, out_features=dim_m)
        self.linear_logvar = nn.Linear(in_features=1000, out_features=dim_m)

        # Distribution for reparam trick
        self.std_normal = torch.distributions.Normal(0, 1)
        if cfg.DEVICE == 'cuda':
            self.std_normal.loc = self.std_normal.loc.cuda()
            self.std_normal.scale = self.std_normal.scale.cuda()

        # Track KL divergence
        self.kl = 0

    def forward(self, x):
        # Encode input
        x = self.input_encoder(x)

        # Reshape to vector
        x = torch.flatten(x, start_dim=1)

        # MLP layers
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))

        # Generate mean and logvar
        mu = self.linear_mu(x)
        sigma = torch.exp(0.5 * self.linear_logvar(x))

        # Reparameterization trick
        m = mu + sigma * self.std_normal.sample(mu.shape)

        self.kl = (
            (sigma ** 2) / 2 +
            (mu ** 2) / 2 -
            torch.log(sigma) -
            1 / 2
        ).sum()

        return m
