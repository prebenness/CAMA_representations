from functools import reduce

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.convnet.cama_convnet import CAMAConvNet

class QZ(nn.Module):
  ''' Parameterises q(z|x,y,m)
  '''
  def __init__(self, dim_z, dim_m, dim_y, x_shape=(1,28,28), num_conv_blocks=3):
    super(QZ, self).__init__()

    self.dim_z = dim_z

    # Convolutional Networks
    self.conv = CAMAConvNet(filter_size=5, in_shape=x_shape, num_blocks=num_conv_blocks)

    # MLP layers
    ## [4*4*64, 500]
    self.linear1 = nn.Linear(in_features=reduce(lambda a, b: a * b, self.conv.out_shape),
                             out_features=4*4*64)
    self.linear2 = nn.Linear(in_features=4*4*64, out_features=500)

    ## After merge with y and m
    in_dim = 500 + dim_y + dim_m
    self.linear3 = nn.Linear(in_features=in_dim, out_features=500)
    self.linear_mu = nn.Linear(in_features=500, out_features=dim_z)
    self.linear_logvar = nn.Linear(in_features=500, out_features=dim_z)

    # Distribution for reparam trick
    self.std_normal = torch.distributions.Normal(0, 1)
    self.std_normal.loc = self.std_normal.loc.cuda()
    self.std_normal.scale = self.std_normal.scale.cuda()

    # Keep track of KL divergence between q(z|x,y,m)||p(z)
    self.kl = 0

  def forward(self, x, y, m):
    # Conv network
    x = self.conv(x)
    
    # Flatten followed by fully connected
    x = torch.flatten(x, start_dim=1)
    x = F.relu(self.linear1(x))
    x = F.relu(self.linear2(x))
    
    # Concat in y and m
    conc = torch.concat((x, y, m), dim=-1)
    conc = F.relu(self.linear3(conc))

    mu = self.linear_mu(conc)
    sigma = torch.exp(0.5 * self.linear_logvar(conc))

    # Reparameterization trick
    z = mu + sigma * self.std_normal.sample(mu.shape)

    # KL-divergence
    ## TODO: if many samples of m are available, should take average KL div
    self.kl = (
      ( sigma ** 2 ) / 2 + 
      ( mu ** 2 ) / 2 - 
      torch.log(sigma) - 
      1 / 2
    ).sum()

    return z