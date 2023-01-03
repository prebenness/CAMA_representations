import torch
import torch.nn as nn

from src.models.single_modality.encoder.qm import QM
from src.models.single_modality.encoder.qz import QZ


class VariationalEncoder(nn.Module):
    def __init__(self, dim_y, dim_z, dim_m, in_shape, device):
        super(VariationalEncoder, self).__init__()

        self.device = device

        self.in_shape = in_shape

        # Set dimensions of hidden state representations
        self.dim_y = dim_y
        self.dim_z = dim_z
        self.dim_m = dim_m

        # Instantiate encoder distributions
        self.qm = QM(dim_m=self.dim_m, in_shape=self.in_shape)
        self.qz = QZ(dim_z=self.dim_z, dim_m=self.dim_m, dim_y=self.dim_y,
                     x_shape=self.in_shape)

    def forward(self, x, y, infer_m=False):
        if infer_m:
            # Sample m ~ q(m|x) unknown
            m = self.qm(x)
        else:
            # Otherwise claim clean sample and set to zero
            m = torch.zeros((x.shape[0], self.dim_m)).to(self.device)

        # Sample z ~ q(z|x,y,m)
        z = self.qz(x, y, m)

        return m, z
