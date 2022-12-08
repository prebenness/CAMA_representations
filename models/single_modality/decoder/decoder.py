import torch.nn as nn

from models.single_modality.decoder.py import PY
from models.single_modality.decoder.pz import PZ
from models.single_modality.decoder.pm import PM
from models.single_modality.decoder.p_merge import PMerge


class Decoder(nn.Module):
  def __init__(self, dim_y, dim_z, dim_m, out_shape):
    super(Decoder, self).__init__()
    
    self.out_shape = out_shape

    # Set dimensions of hidden state representations
    self.dim_y = dim_y
    self.dim_z = dim_z
    self.dim_m = dim_m

    # Embedding networks
    self.py = PY(dim_y=self.dim_y)
    self.pz = PZ(dim_z=self.dim_z)
    self.pm = PM(dim_m=self.dim_m)
    self.p_merge = PMerge()

  def forward(self, y, z, m):
    # Create hidden representations
    h_y = self.py(y)
    h_z = self.pz(z)
    h_m = self.pm(m)

    # Merge
    h = self.p_merge(h_y, h_z, h_m)

    return h