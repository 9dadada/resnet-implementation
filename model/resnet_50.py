import torch
import torch.nn as nn


class Bottleneck(nn.Module):
    """ResNet-50/101/152에서 쓰이는 병목 블록.
    구조: 1x1 conv (채널 축소) → 3x3 conv → 1x1 conv (채널 확장)
    마지막에 입력을 shortcut으로 더해 residual 학습.
    BasicBlock 대비 conv가 3개이고, 3x3은 줄어든 채널에서 수행해 연산량이 작다.
    """
    expansion = 4  # 출력 채널 = out_channels * 4

    def __init__(self, in_channels, out_channels, stride=1):
        super(Bottleneck, self).__init__()

        # 1x1 conv: 채널을 out_channels로 줄임 (병목 시작)
        self.conv1 = nn.Conv2d(in_channels, out_channels,
                               kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)

        # 3x3 conv: 줄어든 채널에서 공간 정보 학습
        self.conv2 = nn.Conv2d(out_channels, out_channels,
                               kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # 1x1 conv: 채널을 4배로 확장 (병목 끝)
        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion,
                               kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)

        self.relu = nn.ReLU(inplace=True)

        # shortcut: 입력과 최종 출력(out_channels*4)의 모양이 다르면 맞춰줌
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels * self.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * self.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * self.expansion),
            )

    def forward(self, x):
        identity = self.shortcut(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += identity  # skip connection
        out = self.relu(out)
        return out


class ResNet50(nn.Module):
    """논문 Table 1의 50-layer ResNet. CIFAR-10용으로 conv1만 3x3 stride 1로 조정.
    블록 구성 (3, 4, 6, 3)은 ResNet-34와 같지만 블록 종류가 Bottleneck.
    최종 특징 채널은 2048 (out_channels 512 * expansion 4).
    """
    def __init__(self, num_classes=1000):
        super(ResNet50, self).__init__()

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        # in_channels는 이전 stage의 출력 채널(= out_channels * 4)
        self.layer1 = self._make_layer(in_channels=64,   out_channels=64,  blocks=3, stride=1)
        self.layer2 = self._make_layer(in_channels=256,  out_channels=128, blocks=4, stride=2)
        self.layer3 = self._make_layer(in_channels=512,  out_channels=256, blocks=6, stride=2)
        self.layer4 = self._make_layer(in_channels=1024, out_channels=512, blocks=3, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(2048, num_classes)  # 512 * expansion(4) = 2048

    def _make_layer(self, in_channels, out_channels, blocks, stride):
        layers = []
        # 첫 블록: stride와 채널 변환 담당
        layers.append(Bottleneck(in_channels, out_channels, stride))
        # 이후 블록: 입력 채널은 out_channels * 4, stride=1
        for _ in range(1, blocks):
            layers.append(Bottleneck(out_channels * Bottleneck.expansion,
                                     out_channels, stride=1))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


if __name__ == "__main__":
    model = ResNet50(num_classes=10)
    dummy_input = torch.randn(1, 3, 32, 32)
    output = model(dummy_input)
    print("ResNet50 output:", output.shape)  # → torch.Size([1, 10])
