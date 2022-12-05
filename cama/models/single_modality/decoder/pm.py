import torch
import torch.nn as nn
import torch.nn.functional as F


class PM(nn.Module):
  def __init__(self, dim_m):
    super().__init__()

    self.dim_m = dim_m

    self.linear1 = nn.Linear(in_features=self.dim_m, out_features=500)
    self.linear2 = nn.Linear(in_features=500, out_features=500)
    self.linear3 = nn.Linear(in_features=500, out_features=500)
    self.linear4 = nn.Linear(in_features=500, out_features=500)

  def forward(self, m):
    m = F.relu(self.linear1(m))
    m = F.relu(self.linear2(m))
    m = F.relu(self.linear3(m))
    m = torch.sigmoid(self.linear4(m))

    return m