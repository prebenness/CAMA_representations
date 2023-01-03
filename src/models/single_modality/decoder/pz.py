import torch
from torch import nn
import torch.nn.functional as F


class PZ(nn.Module):
    def __init__(self, dim_z):
        super().__init__()

        self.dim_z = dim_z

        self.linear1 = nn.Linear(in_features=self.dim_z, out_features=500)
        self.linear2 = nn.Linear(in_features=500, out_features=500)

    def forward(self, z):
        z = F.relu(self.linear1(z))
        z = torch.sigmoid(self.linear2(z))

        return z
