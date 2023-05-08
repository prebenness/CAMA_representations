'''
CNNs as used in the standard CAMA architecture
'''
import torch
from torch import nn
import torch.nn.functional as F
from torchsummary import summary


class CNNMNIST(nn.Module):
    '''
    Convnet as described in "A causal view on the robustness of neural
    networks". 3 blocks of {conv2d(kernel=3), maxpool(kernel=2)}
    All layers have 64 output channels
    '''

    def __init__(self, kernel_size=3):
        super(CNNMNIST, self).__init__()

        assert kernel_size in [3, 5]
        self.kernel_size = kernel_size
        # Set padding so that image dimensions come out correctly
        base_padding = 2 if self.kernel_size == 3 else 3

        # Conv layers
        # (1, 28, 28) -> (64, 30, 30)
        self.conv1 = nn.Conv2d(
            # k=3, p=2 or k=5, p=3
            in_channels=1, out_channels=64, kernel_size=kernel_size,
            padding=base_padding
        )
        # (64, 30, 30) -> (64, 15, 15)
        self.pool1 = nn.MaxPool2d(kernel_size=2)

        # (64, 15, 15) -> (64, 17, 17)
        self.conv2 = nn.Conv2d(
            in_channels=64, out_channels=64, kernel_size=kernel_size,
            padding=base_padding
        )
        # (64, 17, 17) -> (64, 8, 8)
        self.pool2 = nn.MaxPool2d(kernel_size=2)

        # (64, 8, 8) -> (64, 8, 8)
        self.conv3 = nn.Conv2d(
            in_channels=64, out_channels=64, kernel_size=kernel_size,
            padding=base_padding-1
        )
        self.pool3 = nn.MaxPool2d(kernel_size=2)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)

        x = F.relu(self.conv2(x))
        x = self.pool2(x)

        x = F.relu(self.conv3(x))
        x = self.pool3(x)

        return x


class CNNCIFAR(nn.Module):
    '''
    Convnet as described in "A causal view on the robustness of neural
    networks". N blocks of {conv2d(kernel=3), maxpool(kernel=2)}
    All layers have 64 output channels
    '''

    def __init__(self):
        super(CNNCIFAR, self).__init__()

        # Conv layers
        # (3, 32, 32) -> (64, 32, 32)
        self.conv1 = nn.Conv2d(
            in_channels=3, out_channels=64, kernel_size=3, padding=1
        )
        # (64, 32, 32) -> (64, 16, 16)
        self.pool1 = nn.MaxPool2d(kernel_size=2)

        # (64, 16, 16) -> (64, 16, 16)
        self.conv2 = nn.Conv2d(
            in_channels=64, out_channels=64, kernel_size=3, padding=1
        )
        # (64, 16, 16) -> (64, 8, 8)
        self.pool2 = nn.MaxPool2d(kernel_size=2)

        # (64, 8, 8) -> (64, 8, 8)
        self.conv3 = nn.Conv2d(
            in_channels=64, out_channels=64, kernel_size=3, padding=1
        )
        # (64, 8, 8) -> (64, 4, 4)
        self.pool3 = nn.MaxPool2d(kernel_size=2)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)

        x = F.relu(self.conv2(x))
        x = self.pool2(x)

        x = F.relu(self.conv3(x))
        x = self.pool3(x)

        return x


class CNNIMAGENET(nn.Module):
    def __init__(self):
        super(CNNIMAGENET, self).__init__()

        # Conv layers
        # (3, 224, 224) -> (128, 224, 224)
        self.conv1 = nn.Conv2d(
            in_channels=3, out_channels=128, kernel_size=3, padding=1
        )
        # (128, 224, 224) -> (128, 112, 112)
        self.pool1 = nn.MaxPool2d(kernel_size=2)

        # (128, 112, 112) -> (128, 112, 112)
        self.conv2 = nn.Conv2d(
            in_channels=128, out_channels=128, kernel_size=3, padding=1
        )
        # (128, 112, 112) -> (128, 56, 56)
        self.pool2 = nn.MaxPool2d(kernel_size=2)

        # (128, 56, 56) -> (128, 56, 56)
        self.conv3 = nn.Conv2d(
            in_channels=128, out_channels=128, kernel_size=3, padding=1
        )
        # (128, 56, 56) -> (128, 28, 28)
        self.pool3 = nn.MaxPool2d(kernel_size=2)

        # (128, 28, 28) -> (128, 28, 28)
        self.conv4 = nn.Conv2d(
            in_channels=128, out_channels=128, kernel_size=3, padding=1
        )
        # (128, 28, 28) -> (128, 14, 14)
        self.pool4 = nn.MaxPool2d(kernel_size=2)

        # (128, 14, 14) -> (64, 16, 16)
        self.conv5 = nn.Conv2d(
            in_channels=128, out_channels=64, kernel_size=3, padding=2
        )
        # (64, 16, 16) -> (64, 8, 8)
        self.pool5 = nn.MaxPool2d(kernel_size=2)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)

        x = F.relu(self.conv2(x))
        x = self.pool2(x)

        x = F.relu(self.conv3(x))
        x = self.pool3(x)

        x = F.relu(self.conv4(x))
        x = self.pool4(x)

        x = F.relu(self.conv5(x))
        x = self.pool5(x)

        return x


class CNNMedium(nn.Module):
    def __init__(self):
        super(CNNMedium, self).__init__()

        # Conv layers
        # (3, 96, 96) -> (64, 96, 96)
        self.conv1 = nn.Conv2d(
            in_channels=3, out_channels=64, kernel_size=3, padding=1
        )
        # (64, 96, 96) -> (64, 48, 48)
        self.pool1 = nn.MaxPool2d(kernel_size=2)

        # (64, 48, 48) -> (64, 48, 48)
        self.conv2 = nn.Conv2d(
            in_channels=64, out_channels=64, kernel_size=3, padding=1
        )
        # (64, 48, 48) -> (64, 24, 24)
        self.pool2 = nn.MaxPool2d(kernel_size=2)

        # (64, 24, 24) -> (64, 24, 24)
        self.conv3 = nn.Conv2d(
            in_channels=64, out_channels=64, kernel_size=3, padding=1
        )
        # (64, 24, 24) -> (64, 12, 12)
        self.pool3 = nn.MaxPool2d(kernel_size=2)

        # (64, 12, 12) -> (64, 12, 12)
        self.conv4 = nn.Conv2d(
            in_channels=64, out_channels=64, kernel_size=3, padding=1
        )
        # (64, 12, 12) -> (64, 6, 6)
        self.pool4 = nn.MaxPool2d(kernel_size=2)

        # (64, 6, 6) -> (64, 8, 8)
        self.conv5 = nn.Conv2d(
            in_channels=64, out_channels=64, kernel_size=3, padding=2
        )
        # (64, 8, 8) -> (64, 4, 4)
        self.pool5 = nn.MaxPool2d(kernel_size=2)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)

        x = F.relu(self.conv2(x))
        x = self.pool2(x)

        x = F.relu(self.conv3(x))
        x = self.pool3(x)

        x = F.relu(self.conv4(x))
        x = self.pool4(x)

        x = F.relu(self.conv5(x))
        x = self.pool5(x)

        return x


def test():
    '''
    Check that all configurations result in output of shape (-1, 64, 4, 4)
    '''

    batch_size = 128

    x_mnist = torch.rand((batch_size, 1, 28, 28)).to('cuda')
    x_cifar = torch.rand((batch_size, 3, 32, 32)).to('cuda')
    x_imagenet = torch.rand((batch_size, 3, 224, 224)).to('cuda')
    x_pcam = torch.rand((batch_size, 3, 96, 96)).to('cuda')

    m1 = CNNMNIST(kernel_size=3).to('cuda')
    m2 = CNNMNIST(kernel_size=5).to('cuda')
    m3 = CNNCIFAR().to('cuda')
    m4 = CNNIMAGENET().to('cuda')
    m5 = CNNMedium().to('cuda')

    print('MNIST KERNEL=3 #################')
    summary(model=m1, input_size=(1, 28, 28))

    print('MNIST KERNEL=5 #################')
    summary(model=m2, input_size=(1, 28, 28))

    print('CIFAR ##########################')
    summary(model=m3, input_size=(3, 32, 32))

    print('IMAGENET #######################')
    summary(model=m4, input_size=(3, 224, 224))

    print('MEDIUM #########################')
    summary(model=m5, input_size=(3, 96, 96))

    h_x_mnist_1 = m1(x_mnist)
    assert h_x_mnist_1.shape == (batch_size, 64, 4, 4)
    h_x_mnist_2 = m2(x_mnist)
    assert h_x_mnist_2.shape == (batch_size, 64, 4, 4)
    h_x_cifar = m3(x_cifar)
    assert h_x_cifar.shape == (batch_size, 64, 4, 4)
    h_x_imagenet = m4(x_imagenet)
    assert h_x_imagenet.shape == (batch_size, 64, 8, 8)
    h_x_pcam = m5(x_pcam)
    assert h_x_pcam.shape == (batch_size, 64, 4, 4)

    print('All tests completed successfully')


if __name__ == '__main__':
    test()
