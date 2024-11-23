import torch
import torch.nn as nn


class ResNetBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int):
        super(ResNetBlock, self).__init__()

        # Convolutional layer 1
        self.conv1: nn.Conv2d = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=3, stride=stride, padding=1, bias=False)

        # Batch normalization
        self.bn1: nn.BatchNorm2d = nn.BatchNorm2d(out_channels)

        # Convolutional layer 2
        self.conv2: nn.Conv2d = nn.Conv2d(
            out_channels, out_channels,
            kernel_size=3, stride=1, padding=1, bias=False)

        # Batch normalization
        self.bn2: nn.BatchNorm2d = nn.BatchNorm2d(out_channels)

        # ReLU activation function
        self.relu: nn.ReLU = nn.ReLU(inplace=True)

        # Shortcut or skip connection
        self.shortcut: nn.Sequential = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out: torch.Tensor = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.relu(out)
        return out


class ResNet18(nn.Module):
    def __init__(self, num_classes=10):
        super(ResNet18, self).__init__()
        self.in_channels: int = 64
        self.conv1: nn.Conv2d = nn.Conv2d(
            3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1: nn.BatchNorm2d = nn.BatchNorm2d(64)
        self.relu: nn.ReLU = nn.ReLU(inplace=True)
        self.maxpool: nn.MaxPool2d = nn.MaxPool2d(
            kernel_size=3, stride=2, padding=1)

        self.layer1: nn.Sequential = self._make_layer(
            ResNetBlock, 64, 2, stride=1)
        self.layer2: nn.Sequential = self._make_layer(
            ResNetBlock, 128, 2, stride=2)
        self.layer3: nn.Sequential = self._make_layer(
            ResNetBlock, 256, 2, stride=2)
        self.layer4: nn.Sequential = self._make_layer(
            ResNetBlock, 512, 2, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, block, out_channels, num_blocks, stride) -> nn.Sequential:
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out: torch.Tensor = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.maxpool(out)

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out
