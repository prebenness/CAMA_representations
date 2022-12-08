import torch
import torch.nn as nn
import torch.nn.functional as F

from models.single_modality.decoder.decoder import Decoder
from models.single_modality.encoder.variational_encoder import VariationalEncoder

class CAMA(nn.Module):
  def __init__(self, dim_y, dim_z, dim_m, out_shape=(1,28,28), device='cuda'):
    super(CAMA, self).__init__()

    self.device = device

    # Out shape
    self.out_shape = out_shape

    # Set dimensions of hidden state representations
    self.dim_y = dim_y
    self.dim_z = dim_z
    self.dim_m = dim_m

    self.encoder = VariationalEncoder(dim_y=self.dim_y, dim_z=self.dim_z,
                                      dim_m=self.dim_m, in_shape=self.out_shape, device=self.device)
    self.decoder = Decoder(dim_y=self.dim_y, dim_z=self.dim_z, dim_m=self.dim_m,
                           out_shape=self.out_shape)

  def forward(self, x, y, infer_m=False):
    m, z = self.encoder(x, y, infer_m=infer_m)

    x_recon = self.decoder(y, z, m)

    return x_recon

  def predict(self, x):
    # Copy x to get all possible class assignments for each sample
    x_rep = x.repeat(self.dim_y, 1, 1, 1).to(x.device)      # -> [x_1, x_2, ..., x_128, x_1, x_2, ...]
    y_rep = torch.diag(torch.ones(self.dim_y)).to(x.device) # -> [y=1, y=2, ...]
    y_rep = y_rep.repeat_interleave(x.shape[0], dim=0)      # -> [y=1, y=1, ..., y=2, y=2, ...]

    # Sample m
    m = self.encoder.qm(x_rep)

    ## z ~ q(z_k | x, y_c, m)
    # TODO: implement more than one sample of z and take average
    z_k, q_z = self.encoder.qz(x_rep, y_rep, m, return_prob=True)

    # Estimate p(y|x)
    ## Reconstruction probability p(x | y_c, z_k, m)
    p_x = 0.5 * ((self.decoder(y_rep, z_k, m) - x_rep) ** 2).sum(dim=(-1, -2))

    ## Uniform prior class probability p(y_c)
    p_y = (torch.ones((x_rep.shape[0], 1)) * (1 / self.dim_y)).to(x.device)

    ## Prior probability p(z_k):
    ### Diagonal covariance matrix so joint is product of marginals
    p_z = torch.exp(self.encoder.qz.std_normal.log_prob(z_k).sum(-1, keepdim=True))

    ## Pre-softmax class posterior probability
    y_pred_flat = p_x * p_y * p_z / q_z                         # -> [p(y=1|x_1), p(y=1|x_2), ...]

    y_pred = y_pred_flat.reshape((self.dim_y, x.shape[0])).transpose(0, 1)

    y_pred = F.softmax(y_pred, dim=-1)

    return y_pred