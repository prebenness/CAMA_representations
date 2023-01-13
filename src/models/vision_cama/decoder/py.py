from torch import nn
import torch.nn.functional as F


class PYMNIST(nn.Module):
    '''
    MLP with layer dims: [dim_y, 500, 500]
    Relu activations
    '''

    def __init__(self, dim_y):
        super().__init__()

        self.dim_y = dim_y

        h_dim = 500

        self.linear1 = nn.Linear(in_features=self.dim_y, out_features=h_dim)
        self.linear2 = nn.Linear(in_features=h_dim, out_features=h_dim)

    def forward(self, y):
        y = F.relu(self.linear1(y))
        y = F.relu(self.linear2(y))

        return y


class PYCIFAR(nn.Module):
    '''
    MLP with layer dims: [dim_y, 1000, 1000]
    Relu activations
    '''

    def __init__(self, dim_y):
        super().__init__()

        self.dim_y = dim_y

        h_dim = 1000

        self.linear1 = nn.Linear(in_features=self.dim_y, out_features=h_dim)
        self.linear2 = nn.Linear(in_features=h_dim, out_features=h_dim)

    def forward(self, y):
        y = F.relu(self.linear1(y))
        y = F.relu(self.linear2(y))

        return y
