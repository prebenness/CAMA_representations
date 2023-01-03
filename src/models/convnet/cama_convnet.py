'''
Full DeepCAMA model implementation
'''

from torch import nn


class CAMAConvNet(nn.Module):
    '''
    Convnet as described in "A causal view on the robustness of neural
    networks". N blocks of {conv2d(kernel=3), maxpool(kernel=2)}
    All layers have 64 output channels
    '''

    def __init__(self, filter_size=3, in_shape=(1, 28, 28), num_blocks=3):
        super(CAMAConvNet, self).__init__()

        self.filter_size = filter_size
        self.in_shape = in_shape
        self.in_channels = in_shape[0]  # Enforce channels first format
        self.num_blocks = num_blocks
        self.out_shape = self.get_out_shape()

        # Instantiate conv-blocks
        self.convnet = nn.Sequential(
            *self.make_block_stack()
        )

    def forward(self, x):
        return self.convnet(x)

    def get_out_shape(self):
        out_channels = 64
        out_w = self.in_shape[1] // 2 ** self.num_blocks
        out_h = self.in_shape[2] // 2 ** self.num_blocks

        return (out_channels, out_w, out_h)

    def make_block_stack(self):
        # First block
        blocks = [
            nn.Conv2d(kernel_size=self.filter_size, in_channels=self.in_channels,
                      out_channels=64, stride=1, padding='same'),
            nn.MaxPool2d(kernel_size=2),
        ]

        # Generic blocks
        for _ in range(self.num_blocks - 1):
            blocks += [
                nn.Conv2d(kernel_size=self.filter_size, in_channels=64,
                          out_channels=64, stride=1, padding='same'),
                nn.MaxPool2d(kernel_size=2)
            ]

        return nn.Sequential(*blocks)
