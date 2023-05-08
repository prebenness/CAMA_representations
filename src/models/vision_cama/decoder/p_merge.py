'''
P_merge network, which takes the hidden representations of h_y, h_z, and h_m
and attempts to reconstruct the input x
'''

import torch
from torch import nn
import torch.nn.functional as F
from torchsummary import summary


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


class PMergeIMAGENET(nn.Module):
    '''
    Projection MLP: [3000, 8*8*64], followed by 4 deconvolution networks
    with stride 2 and kernel (ideally) 3
    '''

    def __init__(self):
        super().__init__()

        # 1500 + 1500 + 1500 -> 128*4*4
        self.linear = nn.Linear(in_features=4500, out_features=128*4*4)

        self.deconv1 = nn.Sequential(
            # 128,4,4 -> 128,9,9
            nn.ConvTranspose2d(
                in_channels=128, out_channels=128, kernel_size=3, stride=2,
                padding=0
            ),
            nn.ReLU(),
        )

        self.deconv2 = nn.Sequential(
            # 128,9,9 -> 128,17,17
            nn.ConvTranspose2d(
                in_channels=128, out_channels=128, kernel_size=3, stride=2,
                padding=1
            ),
            nn.ReLU(),
        )

        self.deconv3 = nn.Sequential(
            # 128,17,17 -> 128,15,15
            nn.ConvTranspose2d(
                in_channels=128, out_channels=128, kernel_size=3, stride=1,
                padding=2
            ),
            nn.ReLU(),
            # 128,15,15 -> 128,30,30
            nn.Upsample(scale_factor=2),
        )

        self.deconv4 = nn.Sequential(
            # 128,30,30 -> 128,57,57
            nn.ConvTranspose2d(
                in_channels=128, out_channels=128, kernel_size=3, stride=2,
                padding=2
            ),
            nn.ReLU(),
        )

        self.deconv5 = nn.Sequential(
            # 128,57,57 -> 128,111,111
            nn.ConvTranspose2d(
                in_channels=128, out_channels=128, kernel_size=3, stride=2,
                padding=2
            ),
            nn.ReLU(),
            # 128,111,111 -> 128,222,222
            nn.Upsample(scale_factor=2),
        )

        # No non-linearity on final layer
        # 128,222,222 -> 3,224,224
        self.deconv_final = nn.ConvTranspose2d(
            in_channels=128, out_channels=3, kernel_size=3, stride=1,
            padding=0
        )

    def forward(self, y, z, m):
        # Concat representations
        x = torch.concat((y, z, m), dim=-1)

        # Projection
        x = self.linear(F.relu(x))

        # Reshape to 3D tensor
        x = x.reshape((-1, 128, 4, 4))

        # Deconv layers
        x = self.deconv1(x)
        x = self.deconv2(x)
        x = self.deconv3(x)
        x = self.deconv4(x)
        x = self.deconv5(x)

        x = torch.sigmoid(self.deconv_final(x))

        return x


class PMergeMedium(nn.Module):
    '''
    Projection MLP: [3000, 8*8*64], followed by 6 deconvolution networks
    with stride 2 and kernel (ideally) 3
    '''

    def __init__(self):
        super().__init__()

        # 1000 + 1000 + 1000 -> 64*4*4
        self.linear = nn.Linear(in_features=3000, out_features=64*4*4)

        self.deconv1 = nn.Sequential(
            # 64,4,4 -> ,5,5
            nn.ConvTranspose2d(
                in_channels=64, out_channels=64, kernel_size=4, stride=1,
                padding=1,
            ),
            nn.ReLU(),
            # -> ,10,10
            nn.Upsample(scale_factor=2),
        )

        self.deconv2 = nn.Sequential(
            # -> ,11,11
            nn.ConvTranspose2d(
                in_channels=64, out_channels=64, kernel_size=4, stride=1,
                padding=1,
            ),
            nn.ReLU(),
            # -> ,22,22
            nn.Upsample(scale_factor=2),
        )

        self.deconv3 = nn.Sequential(
            # -> ,23,23
            nn.ConvTranspose2d(
                in_channels=64, out_channels=64, kernel_size=4, stride=1,
                padding=1, output_padding=0
            ),
            nn.ReLU(),
            # -> ,46,46
            nn.Upsample(scale_factor=2),
        )

        self.deconv4 = nn.Sequential(
            # -> ,48,48
            nn.ConvTranspose2d(
                in_channels=64, out_channels=64, kernel_size=3, stride=1,
                padding=0
            ),
            nn.ReLU(),
            # -> ,96,96
            nn.Upsample(scale_factor=2),
        )

        self.deconv5 = nn.Sequential(
            # -> ,94,94
            nn.ConvTranspose2d(
                in_channels=64, out_channels=64, kernel_size=3, stride=1,
                padding=2
            ),
            nn.ReLU(),
        )

        # No non-linearity on final layer
        # -> ,96,96
        self.deconv_final = nn.ConvTranspose2d(
            in_channels=64, out_channels=3, kernel_size=3, stride=1,
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
        x = self.deconv1(x)
        x = self.deconv2(x)
        x = self.deconv3(x)
        x = self.deconv4(x)
        x = self.deconv5(x)

        x = torch.sigmoid(self.deconv_final(x))

        return x


def test():
    m1 = PMergeIMAGENET().to('cuda')
    summary(model=m1, input_size=[(1500,), (1500,), (1500,)])

    m2 = PMergeMedium().to('cuda')
    summary(model=m2, input_size=[(1000,), (1000,), (1000,)])


if __name__ == '__main__':
    test()
