from functools import reduce

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.convnet.cama_convnet import CAMAConvNet


class QM(nn.Module):
  ''' Parameterises q(m|x)
  '''
  def __init__(self, dim_m, in_shape=(1,28,28), num_conv_blocks=3):
    super(QM, self).__init__()

    self.dim_m = dim_m

    # Convolutional Networks
    self.conv = CAMAConvNet(filter_size=3, in_shape=in_shape, num_blocks=num_conv_blocks)
    
    # MLP layers
    ## [ 4*4*64, 500, dim_m*2 ]
    self.linear1 = nn.Linear(in_features=reduce(lambda a, b: a * b, self.conv.out_shape),
                             out_features=4*4*64)
    self.linear2 = nn.Linear(in_features=4*4*64, out_features=500)
    self.linear_mu = nn.Linear(in_features=500, out_features=dim_m)
    self.linear_logvar = nn.Linear(in_features=500, out_features=dim_m)

    # Distribution for reparam trick
    self.std_normal = torch.distributions.Normal(0, 1)
    self.std_normal.loc = self.std_normal.loc.cuda()
    self.std_normal.scale = self.std_normal.scale.cuda()

    # Track KL divergence
    self.kl = 0

  def forward(self, x):
    x = self.conv(x)
    x = torch.flatten(x, start_dim=1)

    x = F.relu(self.linear1(x))
    x = F.relu(self.linear2(x))
    
    mu = self.linear_mu(x)
    sigma = torch.exp(0.5 * self.linear_logvar(x))

    # Reparameterization trick
    m = mu + sigma * self.std_normal.sample(mu.shape)

    self.kl = (
      ( sigma ** 2 ) / 2 + 
      ( mu ** 2 ) / 2 - 
      torch.log(sigma) - 
      1 / 2
    ).sum()

    return m