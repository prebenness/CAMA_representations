from functools import reduce

import torch
import torch.nn as nn
import torch.nn.functional as F


class VariationalEncoder(nn.Module):
  def __init__(self, in_dims, latent_dim, hdim=512):
    super(VariationalEncoder, self).__init__()

    self.in_dims = in_dims
    self.latent_dim = latent_dim

    self.hdim = hdim
    self.linear1 = nn.Linear(in_features=reduce(lambda a, b: a * b, self.in_dims), out_features=self.hdim)
    self.linear2a = nn.Linear(in_features=self.hdim, out_features=self.latent_dim)
    self.linear2b = nn.Linear(in_features=self.hdim, out_features=self.latent_dim)

    self.N = torch.distributions.Normal(0, 1)
    self.N.loc = self.N.loc.cuda()
    self.N.scale = self.N.scale.cuda()

    self.kl = 0

  def forward(self, x):
    x = torch.flatten(x, start_dim=1)
    x = F.relu(self.linear1(x))
    
    mu = self.linear2a(x)
    sigma = torch.exp(self.linear2b(x))

    z = mu + sigma * self.N.sample(mu.shape)

    self.kl = (sigma ** 2 + mu ** 2 - torch.log(sigma) - 1/2).sum()

    return z


class Decoder(nn.Module):
  def __init__(self, latent_dim, out_dims, hdim=512):
    super(Decoder, self).__init__()

    self.latent_dim = latent_dim
    self.out_dims = out_dims
    self.hdim = hdim

    self.linear1 = nn.Linear(in_features=latent_dim, out_features=self.hdim)
    self.linear2 = nn.Linear(in_features=self.hdim, out_features=reduce(lambda a, b: a * b, self.out_dims))

  def forward(self, z):
    z = F.relu(self.linear1(z))
    z = torch.sigmoid(self.linear2(z))
    z = z.reshape([-1, *self.out_dims])
    return z


class VariationalAutoEncoder(nn.Module):
  def __init__(self, latent_dim, in_dims, hdim=512):
    super(VariationalAutoEncoder, self).__init__()

    self.latent_dim = latent_dim
    self.in_dims = in_dims
    self.hdim=hdim

    self.encoder = VariationalEncoder(in_dims=self.in_dims, latent_dim=self.latent_dim, hdim=self.hdim)
    self.decoder = Decoder(latent_dim=self.latent_dim, out_dims=self.in_dims, hdim=self.hdim)

  def forward(self, x):
    z = self.encoder(x)
    x_rec = self.decoder(z)

    return x_rec