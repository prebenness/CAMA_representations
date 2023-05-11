import torch
import torch.nn as nn
import torch.nn.functional as F

import src.utils.config as cfg
from src.models.vision_cama.encoder.input_encoders.cnns import\
    CNNMNIST, CNNCIFAR, CNNIMAGENET, CNNMedium
from src.models.vision_cama.encoder.input_encoders.resnet import ResNetEmbedder


class QZMNIST(nn.Module):
    ''' Parameterises q(z|x,y,m)
    '''

    def __init__(self, dim_y, dim_z, dim_m):
        super(QZMNIST, self).__init__()

        self.dim_z = dim_z

        # Input encoder
        self.input_encoder = CNNMNIST(kernel_size=5)

        # MLP layers
        # [4*4*64, 500]
        self.linear1 = nn.Linear(in_features=4*4*64, out_features=500)

        # After merge with y and m
        # [500+dim_y+dim_m, 500, dim_z*2]
        self.linear2 = nn.Linear(in_features=500+dim_y+dim_m, out_features=500)
        self.linear_mu = nn.Linear(in_features=500, out_features=dim_z)
        self.linear_logvar = nn.Linear(in_features=500, out_features=dim_z)

        # Distribution for reparam trick
        self.std_normal = torch.distributions.Normal(0, 1)
        if cfg.DEVICE == 'cuda':
            self.std_normal.loc = self.std_normal.loc.cuda()
            self.std_normal.scale = self.std_normal.scale.cuda()

        # Keep track of KL divergence between q(z|x,y,m)||p(z)
        self.kl = 0

    def forward(self, x, y, m, return_log_prob=False):
        # Input encoder
        x = self.input_encoder(x)

        # Flatten followed by fully connected
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.linear1(x))

        # Concat in y and m
        conc = torch.concat((x, y, m), dim=-1)
        conc = F.relu(self.linear2(conc))

        mu = self.linear_mu(conc)
        sigma = torch.exp(0.5 * self.linear_logvar(conc))

        # Reparameterization trick
        z = mu + sigma * self.std_normal.sample(mu.shape)

        # KL-divergence
        # TODO: if many samples of m are available, should take average KL div
        self.kl = (
            (sigma ** 2) / 2 +
            (mu ** 2) / 2 -
            torch.log(sigma) -
            1 / 2
        ).sum()

        if return_log_prob:
            # Return z and q(z|)
            z_std = (z - mu) / sigma
            # Diagonal covariance matrix so joint is product of marginals
            log_prob = self.std_normal.log_prob(
                z_std).type(torch.float64).sum(-1, keepdim=True)

            assert torch.all(torch.isfinite(log_prob))

            return z, log_prob
        else:
            return z


class QZCIFAR(nn.Module):
    ''' Parameterises q(z|x,y,m)
    '''

    def __init__(self, dim_y, dim_z, dim_m):
        super(QZCIFAR, self).__init__()

        self.dim_z = dim_z

        # Input encoder
        self.input_encoder = CNNCIFAR()

        # MLP layers
        # After merge with y and m
        # [64*4*4+dim_y+dim_m, 1000, 1000, dim_z*2]
        self.linear1 = nn.Linear(
            in_features=4*4*64+dim_y+dim_m, out_features=1000
        )
        self.linear2 = nn.Linear(in_features=1000, out_features=1000)
        self.linear_mu = nn.Linear(in_features=1000, out_features=dim_z)
        self.linear_logvar = nn.Linear(in_features=1000, out_features=dim_z)

        # Distribution for reparam trick
        self.std_normal = torch.distributions.Normal(0, 1)
        if cfg.DEVICE == 'cuda':
            self.std_normal.loc = self.std_normal.loc.cuda()
            self.std_normal.scale = self.std_normal.scale.cuda()

        # Keep track of KL divergence between q(z|x,y,m)||p(z)
        self.kl = 0

    def forward(self, x, y, m, return_log_prob=False):
        # Input encoder
        x = self.input_encoder(x)

        # Flatten to vector
        x = torch.flatten(x, start_dim=1)

        # Concat in y and m, and apply MLPs
        conc = torch.concat((x, y, m), dim=-1)
        conc = F.relu(self.linear1(conc))
        conc = F.relu(self.linear2(conc))

        mu = self.linear_mu(conc)
        sigma = torch.exp(0.5 * self.linear_logvar(conc))

        # Reparameterization trick
        z = mu + sigma * self.std_normal.sample(mu.shape)

        # KL-divergence
        # TODO: if many samples of m are available, should take average KL div
        self.kl = (
            (sigma ** 2) / 2 +
            (mu ** 2) / 2 -
            torch.log(sigma) -
            1 / 2
        ).sum()

        if return_log_prob:
            # Return z and q(z|)
            z_std = (z - mu) / sigma
            # Diagonal covariance matrix so joint is product of marginals
            log_prob = self.std_normal.log_prob(
                z_std).type(torch.float64).sum(-1, keepdim=True)

            assert torch.all(torch.isfinite(log_prob))

            return z, log_prob
        else:
            return z


class QZIMAGENET(nn.Module):
    ''' Parameterises q(z|x,y,m)
    '''

    def __init__(self, dim_y, dim_z, dim_m):
        super(QZIMAGENET, self).__init__()

        self.dim_z = dim_z

        # Input encoder
        self.input_encoder = CNNIMAGENET()

        # MLP layers
        # After merge with y and m
        # [64*4*4+dim_y+dim_m, 1000, 1000, dim_z*2]
        self.linear1 = nn.Linear(
            in_features=8*8*64+dim_y+dim_m, out_features=1000
        )
        self.linear2 = nn.Linear(in_features=1000, out_features=1000)
        self.linear_mu = nn.Linear(in_features=1000, out_features=dim_z)
        self.linear_logvar = nn.Linear(in_features=1000, out_features=dim_z)

        # Distribution for reparam trick
        self.std_normal = torch.distributions.Normal(0, 1)
        if cfg.DEVICE == 'cuda':
            self.std_normal.loc = self.std_normal.loc.cuda()
            self.std_normal.scale = self.std_normal.scale.cuda()

        # Keep track of KL divergence between q(z|x,y,m)||p(z)
        self.kl = 0

    def forward(self, x, y, m, return_log_prob=False):
        # Input encoder
        x = self.input_encoder(x)

        # Flatten to vector
        x = torch.flatten(x, start_dim=1)

        # Concat in y and m, and apply MLPs
        conc = torch.concat((x, y, m), dim=-1)
        conc = F.relu(self.linear1(conc))
        conc = F.relu(self.linear2(conc))

        mu = self.linear_mu(conc)
        sigma = torch.exp(0.5 * self.linear_logvar(conc))

        # Reparameterization trick
        z = mu + sigma * self.std_normal.sample(mu.shape)

        # KL-divergence
        # TODO: if many samples of m are available, should take average KL div
        self.kl = (
            (sigma ** 2) / 2 +
            (mu ** 2) / 2 -
            torch.log(sigma) -
            1 / 2
        ).sum()

        if return_log_prob:
            # Return z and q(z|)
            z_std = (z - mu) / sigma
            # Diagonal covariance matrix so joint is product of marginals
            log_prob = self.std_normal.log_prob(
                z_std).type(torch.float64).sum(-1, keepdim=True)

            assert torch.all(torch.isfinite(log_prob))

            return z, log_prob
        else:
            return z


class QZRESNET(nn.Module):
    ''' Parameterises q(z|x,y,m)
    '''

    def __init__(self, dim_y, dim_z, dim_m):
        super(QZRESNET, self).__init__()

        self.dim_z = dim_z

        # Input encoder
        self.input_encoder = ResNetEmbedder(
            resnet_type=18, num_channels=cfg.OUT_SHAPE[0])

        # MLP layers
        # After merge with y and m
        # [512+dim_y+dim_m, 1000, 1000, dim_z*2]
        self.linear1 = nn.Linear(
            in_features=512+dim_y+dim_m, out_features=1000
        )
        self.linear2 = nn.Linear(in_features=1000, out_features=1000)
        self.linear_mu = nn.Linear(in_features=1000, out_features=dim_z)
        self.linear_logvar = nn.Linear(in_features=1000, out_features=dim_z)

        # Distribution for reparam trick
        self.std_normal = torch.distributions.Normal(0, 1)
        if cfg.DEVICE == 'cuda':
            self.std_normal.loc = self.std_normal.loc.cuda()
            self.std_normal.scale = self.std_normal.scale.cuda()

        # Keep track of KL divergence between q(z|x,y,m)||p(z)
        self.kl = 0

    def forward(self, x, y, m, return_log_prob=False):
        # Input encoder
        x = self.input_encoder(x)

        # Flatten to vector
        x = torch.flatten(x, start_dim=1)

        # Concat in y and m, and apply MLPs
        conc = torch.concat((x, y, m), dim=-1)
        conc = F.relu(self.linear1(conc))
        conc = F.relu(self.linear2(conc))

        mu = self.linear_mu(conc)
        sigma = torch.exp(0.5 * self.linear_logvar(conc))

        # Reparameterization trick
        z = mu + sigma * self.std_normal.sample(mu.shape)

        # KL-divergence
        # TODO: if many samples of m are available, should take average KL div
        self.kl = (
            (sigma ** 2) / 2 +
            (mu ** 2) / 2 -
            torch.log(sigma) -
            1 / 2
        ).sum()

        if return_log_prob:
            # Return z and q(z|)
            z_std = (z - mu) / sigma
            # Diagonal covariance matrix so joint is product of marginals
            log_prob = self.std_normal.log_prob(
                z_std).type(torch.float64).sum(-1, keepdim=True)

            assert torch.all(torch.isfinite(log_prob))

            return z, log_prob
        else:
            return z


class QZMedium(nn.Module):
    ''' Parameterises q(z|x,y,m)
    '''

    def __init__(self, dim_y, dim_z, dim_m):
        super(QZMedium, self).__init__()

        self.dim_z = dim_z

        # Input encoder
        self.input_encoder = CNNMedium()

        # MLP layers
        # After merge with y and m
        # [64*4*4+dim_y+dim_m, 1000, 1000, dim_z*2]
        self.linear1 = nn.Linear(
            in_features=4*4*64+dim_y+dim_m, out_features=1000
        )
        self.linear2 = nn.Linear(in_features=1000, out_features=1000)
        self.linear_mu = nn.Linear(in_features=1000, out_features=dim_z)
        self.linear_logvar = nn.Linear(in_features=1000, out_features=dim_z)

        # Distribution for reparam trick
        self.std_normal = torch.distributions.Normal(0, 1)
        if cfg.DEVICE == 'cuda':
            self.std_normal.loc = self.std_normal.loc.cuda()
            self.std_normal.scale = self.std_normal.scale.cuda()

        # Keep track of KL divergence between q(z|x,y,m)||p(z)
        self.kl = 0

    def forward(self, x, y, m, return_log_prob=False):
        # Input encoder
        x = self.input_encoder(x)

        # Flatten to vector
        x = torch.flatten(x, start_dim=1)

        # Concat in y and m, and apply MLPs
        conc = torch.concat((x, y, m), dim=-1)
        conc = F.relu(self.linear1(conc))
        conc = F.relu(self.linear2(conc))

        mu = self.linear_mu(conc)
        sigma = torch.exp(0.5 * self.linear_logvar(conc))

        # Reparameterization trick
        z = mu + sigma * self.std_normal.sample(mu.shape)

        # KL-divergence
        # TODO: if many samples of m are available, should take average KL div
        self.kl = (
            (sigma ** 2) / 2 +
            (mu ** 2) / 2 -
            torch.log(sigma) -
            1 / 2
        ).sum()

        if return_log_prob:
            # Return z and q(z|)
            z_std = (z - mu) / sigma
            # Diagonal covariance matrix so joint is product of marginals
            log_prob = self.std_normal.log_prob(
                z_std).type(torch.float64).sum(-1, keepdim=True)

            assert torch.all(torch.isfinite(log_prob))

            return z, log_prob
        else:
            return z
