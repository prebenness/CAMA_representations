import torch
from torch import nn
import torch.nn.functional as F

import src.utils.config as cfg


class PMerge(nn.Module):
    def __init__(self):
        super().__init__()

        # 500 + 500 + 500 -> 4*4*64
        self.linear = nn.Linear(in_features=1500, out_features=4*4*64)

        # Hard code layouts for MNIST and CIFAR for now
        if cfg.DATASET == 'mnist':
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

            self.deconv = torch.nn.Sequential(
                self.deconv1, self.deconv2, self.deconv3
            )

        elif cfg.DATASET in ['cifar10', 'cifar100']:
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
        else:
            raise NotImplementedError('Only support for MNIST and CIFAR')

    def forward(self, y, z, m):
        # Concat representations
        h = torch.concat((y, z, m), dim=-1)
        h = self.linear(F.relu(h))

        # Reshape to 3D tensor
        h = h.reshape((-1, 64, 4, 4))

        # Deconv layers
        h = self.deconv(h)

        # Tanh final output and ensure in range [-2, 2] as this is bound on
        # standardised pixel values
        h = 2 * torch.tanh(h)

        return h
