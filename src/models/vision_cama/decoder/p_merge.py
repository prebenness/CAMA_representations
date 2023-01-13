'''
P_merge network, which takes the hidden representations of h_y, h_z, and h_m
and attempts to reconstruct the input x
'''

import torch
from torch import nn
import torch.nn.functional as F


class PMergeMNIST(nn.Module):
    '''
    Projection MLP: [1500, 4*4*64], followed by 3 deconvolution networks
    with stride 2 and kernel (ideally) 3
    '''

    def __init__(self):
        super().__init__()

        # 500 + 500 + 500 -> 4*4*64
        self.linear = nn.Linear(in_features=1500, out_features=4*4*64)

        # 4*4*64 -> 64,7,7
        self.deconv1 = nn.ConvTranspose2d(
            in_channels=64, out_channels=64, kernel_size=3, stride=2,
            padding=1
        )

        # 64,7,7 -> 64,13,13
        self.deconv2 = nn.ConvTranspose2d(
            in_channels=64, out_channels=64, kernel_size=3, stride=2,
            padding=1
        )

        # 64,13,13 -> 1,28,28
        self.deconv3 = nn.ConvTranspose2d(
            in_channels=64, out_channels=1, kernel_size=4, stride=2,
            padding=0
        )

    def forward(self, y, z, m):
        # Concat representations
        x = torch.concat((y, z, m), dim=-1)

        # Projection
        x = self.linear(F.relu(x))

        # Reshape to 3D tensor
        x = x.reshape((-1, 64, 4, 4))

        # Deconv layers
        x = F.relu(self.deconv1(x))
        x = F.relu(self.deconv2(x))
        x = torch.sigmoid(self.deconv3(x))

        return x


class PMergeCIFAR(nn.Module):
    '''
    Projection MLP: [3000, 4*4*64], followed by 4 deconvolution networks
    with stride 2 and kernel (ideally) 3
    '''

    def __init__(self):
        super().__init__()

        # 1000 + 1000 + 1000 -> 4*4*64
        self.linear = nn.Linear(in_features=3000, out_features=4*4*64)

        # 4*4*64 -> 64,5,5
        self.deconv1 = nn.ConvTranspose2d(
            in_channels=64, out_channels=64, kernel_size=3, stride=2,
            padding=2
        )

        # 64,5,5 -> 64,9,9
        self.deconv2 = nn.ConvTranspose2d(
            in_channels=64, out_channels=64, kernel_size=3, stride=2,
            padding=1
        )

        # 64,9,9 -> 3,15,15
        self.deconv3 = nn.ConvTranspose2d(
            in_channels=64, out_channels=64, kernel_size=3, stride=2,
            padding=2
        )

        # 64,15,15 -> 3,32,32
        self.deconv4 = nn.ConvTranspose2d(
            in_channels=64, out_channels=3, kernel_size=4, stride=2,
            padding=0
        )

        self.deconv = torch.nn.Sequential(
            self.deconv1, self.deconv2, self.deconv3, self.deconv4
        )

    def forward(self, y, z, m):
        # Concat representations
        x = torch.concat((y, z, m), dim=-1)

        # Projection
        x = self.linear(F.relu(x))

        # Reshape to 3D tensor
        x = x.reshape((-1, 64, 4, 4))

        # Deconv layers
        x = F.relu(self.deconv1(x))
        x = F.relu(self.deconv2(x))
        x = F.relu(self.deconv3(x))
        x = torch.sigmoid(self.deconv4(x))

        return x
