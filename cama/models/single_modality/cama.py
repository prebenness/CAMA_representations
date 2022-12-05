import torch.nn as nn

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