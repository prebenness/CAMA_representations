from torch import nn
import torch.nn.functional as F


class PZMNIST(nn.Module):
    '''
    MLP with layer dims: [dim_z, 500, 500]
    Relu activations
    '''

    def __init__(self, dim_z):
        super().__init__()

        self.dim_z = dim_z

        h_dim = 500

        self.linear1 = nn.Linear(in_features=self.dim_z, out_features=h_dim)
        self.linear2 = nn.Linear(in_features=h_dim, out_features=h_dim)

    def forward(self, z):
        '''
        Embed z -> h_z
        '''
        z = F.relu(self.linear1(z))
        z = F.relu(self.linear2(z))

        return z


class PZCIFAR(nn.Module):
    '''
    MLP with layer dims: [dim_z, 1000, 1000]
    Relu activations
    '''

    def __init__(self, dim_z):
        super().__init__()

        self.dim_z = dim_z

        h_dim = 1000

        self.linear1 = nn.Linear(in_features=self.dim_z, out_features=h_dim)
        self.linear2 = nn.Linear(in_features=h_dim, out_features=h_dim)

    def forward(self, z):
        '''
        Embed z -> h_z
        '''
        z = F.relu(self.linear1(z))
        z = F.relu(self.linear2(z))

        return z


class PZIMAGENET(nn.Module):
    '''
    MLP with layer dims: [dim_z, 1500, 1500, 1500]
    Relu activations
    '''

    def __init__(self, dim_z):
        super().__init__()

        self.dim_z = dim_z

        h_dim = 1500

        self.linear1 = nn.Linear(in_features=self.dim_z, out_features=h_dim)
        self.linear2 = nn.Linear(in_features=h_dim, out_features=h_dim),
        self.linear3 = nn.Linear(in_features=h_dim, out_features=h_dim)

    def forward(self, z):
        '''
        Embed z -> h_z
        '''
        z = F.relu(self.linear1(z))
        z = F.relu(self.linear2(z))
        z = F.relu(self.linear3(z))

        return z
