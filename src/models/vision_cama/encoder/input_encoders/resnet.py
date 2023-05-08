'''
Standard ResNet implementations
'''

import torch
from torch import nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    '''
    ResNet basic block
    '''
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    '''
    ResNet bottleneck
    '''
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion *
                               planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, num_channels=3):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(
            num_channels, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

    def get_rep(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        return out


class ResNetEmbedder(nn.Module):
    def __init__(self, resnet_type, *args, num_channels=3, **kwargs):
        super().__init__(*args, **kwargs)

        if resnet_type == 18:
            num_blocks = [2, 2, 2, 2]
        elif resnet_type == 34:
            num_blocks = [3, 4, 6, 3]
        elif resnet_type == 50:
            num_blocks = [3, 4, 6, 3]
        elif resnet_type == 101:
            num_blocks = [3, 4, 23, 3]
        elif resnet_type == 152:
            num_blocks = [3, 8, 36, 3]
        else:
            supported_types = [18, 34, 50, 101, 152]
            raise NotImplementedError(
                f'resnet_type {resnet_type} must be in {supported_types}'
            )

        self.resnet = ResNet(
            block=BasicBlock, num_blocks=num_blocks,
            num_classes=1, num_channels=num_channels
        )

    def forward(self, x):
        return self.resnet.get_rep(x)


def test():
    batch_size = 10
    shape_r = (batch_size, 512)

    # ResNet18 with CIFAR10 format data
    net18_cifar10 = ResNetEmbedder(resnet_type=18).to('cuda')
    x_cifar10 = torch.randn(1, 3, 32, 32).to('cuda')
    r_cifar10 = net18_cifar10(x_cifar10)
    assert r_cifar10.shape == shape_r

    # ResNet18 with MNIST format data
    net18_mnist = ResNetEmbedder(resnet_type=18, num_channels=1)
    r_mnist = net18_mnist(torch.randn(1, 1, 28, 28))
    assert r_mnist.shape == shape_r


if __name__ == '__main__':
    test()
