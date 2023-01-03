import torch
from torch import nn
import torch.nn.functional as F


class PMerge(nn.Module):
    def __init__(self):
        super().__init__()

        # 500 + 500 + 500 -> 4*4*64 -> deconv
        self.linear = nn.Linear(in_features=1500, out_features=4*4*64)
        self.deconv1 = nn.ConvTranspose2d(in_channels=64, out_channels=64,
                                          kernel_size=3, stride=2, padding=1)
        self.deconv2 = nn.ConvTranspose2d(in_channels=64, out_channels=64,
                                          kernel_size=3, stride=2, padding=1)
        self.deconv3 = nn.ConvTranspose2d(in_channels=64, out_channels=1,
                                          kernel_size=4, stride=2, padding=0)

    def forward(self, y, z, m):
        # Concat representations
        h = torch.concat((y, z, m), dim=-1)
        h = self.linear(F.relu(h))

        # Reshape to 3D tensor
        h = h.reshape((-1, 64, 4, 4))

        # Deconv layers
        h = self.deconv1(h)
        h = self.deconv2(h)
        h = self.deconv3(h)

        # Sigmoid final output
        h = torch.sigmoid(h)

        return h
