'''
ResNeSt
'''
import torch
import torch.nn as nn

from layers import ConvBlock
from layers import GlobalAvgPool2d
from layers import BottleneckBlock

class ResNeSt(nn.Module):
    '''
    ResNeSt [1] class

    [1] ResNeSt : Split-Attention Networks,
        Hang Zhang, Chongruo Wu, Zhongyue Zhang, Yi Zhu, Zhi Zhang, Haibin Lin, Yue Sun, Tong He, Jonas Mueller, R. Manmatha, Mu Li, Alexander Smola,
        https://arxiv.org/abs/2004.08955

    official implementation : https://github.com/zhanghang1989/ResNeSt

    '''
    def __init__(self,
        layers,
        radix=2,
        groups=1,
        bottleneck_width=64,
        n_classes=1000,
        stem_width=64
    ):
        super(ResNeSt, self).__init__()
        self.radix = radix
        self.groups = groups
        self.bottleneck_width = bottleneck_width

        self.deep_stem = nn.Sequential(
            ConvBlock(
                in_channels=3,
                out_channels=stem_width,
                kernel_size=3,
                stride=2,
                padding=1
            ),
            ConvBlock(
                in_channels=stem_width,
                out_channels=stem_width,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            ConvBlock(
                in_channels=stem_width,
                out_channels=stem_width*2,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.MaxPool2d(
                kernel_size=3,
                stride=2,
                padding=1
            )
        )

        self.in_channels = stem_width*2

        self.layer1 = self._make_layers(
            channels=64,
            blocks=layers[0],
            stride=1,
            is_first=False
        )
        self.layer2 = self._make_layers(
            channels=128,
            blocks=layers[1],
            stride=2
        )
        self.layer3 = self._make_layers(
            channels=256,
            blocks=layers[2],
            stride=2
        )
        self.layer4 = self._make_layers(
            channels=512,
            blocks=layers[3],
            stride=2
        )

        self.classifier = nn.Sequential(
            GlobalAvgPool2d(),
            nn.Linear(
                in_features=512*BottleneckBlock.expansion,
                out_features=n_classes
            )
        )


    def _make_layers(self,
        channels,
        blocks,
        stride=1,
        is_first=True
    ):
        down_layers = None
        if not stride ==1 or not self.in_channels == channels * BottleneckBlock.expansion:
            down_layers = nn.Sequential(
                nn.AvgPool2d(
                    kernel_size=stride,
                    stride=stride,
                    ceil_mode=True,
                    count_include_pad=False
                ),
                nn.Conv2d(
                    in_channels=self.in_channels,
                    out_channels=channels*BottleneckBlock.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False
                ),
                nn.BatchNorm2d(channels*BottleneckBlock.expansion)
            )

        layers = []
        layers.append(
            BottleneckBlock(
                in_channels=self.in_channels,
                channels=channels,
                stride=stride,
                downsample=down_layers,
                radix=self.radix,
                groups=self.groups,
                bottleneck_width=self.bottleneck_width,
                is_first=is_first
            )
        )

        self.in_channels = channels * BottleneckBlock.expansion
        for _ in range(1, blocks):
            layers.append(
                BottleneckBlock(
                    in_channels=self.in_channels,
                    channels=channels,
                    radix=self.radix,
                    groups=self.groups,
                    bottleneck_width=self.bottleneck_width
                )
            )

        return nn.Sequential(*layers)

    def forward(self, img):
        x = self.deep_stem(img)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.classifier(x)

        return x

if __name__ == "__main__":
    m = ResNeSt(
        [3, 4, 6, 3]
    )
    img = torch.randn(3, 3, 224, 224)
    print(m(img).size())