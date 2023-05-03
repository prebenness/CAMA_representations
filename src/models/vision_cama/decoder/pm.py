'''
Neural Network for creating a hidden state representation h_m = f(m)
where m ~ q(m|x)
'''
from torch import nn
import torch.nn.functional as F


class PMMNIST(nn.Module):
    '''
    MLP with layer dims: [dim_m, 500, 500, 500, 500]
    ReLu activations
    '''

    def __init__(self, dim_m):
        super().__init__()

        self.dim_m = dim_m

        h_dim = 500

        self.linear1 = nn.Linear(in_features=self.dim_m, out_features=h_dim)
        self.linear2 = nn.Linear(in_features=h_dim, out_features=h_dim)
        self.linear3 = nn.Linear(in_features=h_dim, out_features=h_dim)
        self.linear4 = nn.Linear(in_features=h_dim, out_features=h_dim)

    def forward(self, m):
        '''
        Embed m -> h_m
        '''
        m = F.relu(self.linear1(m))
        m = F.relu(self.linear2(m))
        m = F.relu(self.linear3(m))
        m = F.relu(self.linear4(m))

        return m


class PMCIFAR(nn.Module):
    '''
    MLP with layer dims: [dim_m, 1000, 1000, 1000]
    ReLu activations
    '''

    def __init__(self, dim_m):
        super().__init__()

        self.dim_m = dim_m

        h_dim = 1000

        self.linear1 = nn.Linear(in_features=self.dim_m, out_features=h_dim)
        self.linear2 = nn.Linear(in_features=h_dim, out_features=h_dim)
        self.linear3 = nn.Linear(in_features=h_dim, out_features=h_dim)

    def forward(self, m):
        '''
        Embed m -> h_m
        '''
        m = F.relu(self.linear1(m))
        m = F.relu(self.linear2(m))
        m = F.relu(self.linear3(m))

        return m


class PMIMAGENET(nn.Module):
    '''
    MLP with layer dims: [dim_m, 1500, 1500, 1500, 1500]
    ReLu activations
    '''

    def __init__(self, dim_m):
        super().__init__()

        self.dim_m = dim_m

        h_dim = 1500

        self.linear1 = nn.Linear(in_features=self.dim_m, out_features=h_dim)
        self.linear2 = nn.Linear(in_features=h_dim, out_features=h_dim)
        self.linear3 = nn.Linear(in_features=h_dim, out_features=h_dim),
        self.linear4 = nn.Linear(in_features=h_dim, out_features=h_dim)

    def forward(self, m):
        '''
        Embed m -> h_m
        '''
        m = F.relu(self.linear1(m))
        m = F.relu(self.linear2(m))
        m = F.relu(self.linear3(m))
        m = F.relu(self.linear4(m))

        return m
