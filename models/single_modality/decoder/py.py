import torch
import torch.nn as nn
import torch.nn.functional as F


class PY(nn.Module):
  def __init__(self, dim_y):
    super().__init__()

    self.dim_y = dim_y

    self.linear1 = nn.Linear(in_features=self.dim_y, out_features=500)
    self.linear2 = nn.Linear(in_features=500, out_features=500)

  def forward(self, y):
    y = F.relu(self.linear1(y))
    y = torch.sigmoid(self.linear2(y))

    return y