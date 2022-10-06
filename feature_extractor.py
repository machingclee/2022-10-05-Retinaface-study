import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
import config
from device import device
from torchsummary import summary


class Resnet50FPNFeactureExtractor(nn.Module):
    def __init__(self):
        self.fpn_out_channels = config.fpn_out_channels
        super(Resnet50FPNFeactureExtractor, self).__init__()
        self.resnet50 = models.resnet50(pretrained=True).to(device)

        self.conv2 = nn.Sequential(
            self.resnet50.conv1,
            self.resnet50.bn1,
            self.resnet50.relu,
            self.resnet50.maxpool,
            self.resnet50.layer1
        )
        self.conv3 = self.resnet50.layer2
        self.conv4 = self.resnet50.layer3
        self.conv5 = self.resnet50.layer4

        self.lateral_conv5 = nn.Conv2d(2048, config.fpn_out_channels, 1, 1).to(device)
        self.lateral_conv4 = nn.Conv2d(1024, config.fpn_out_channels, 1, 1).to(device)
        self.lateral_conv3 = nn.Conv2d(512, config.fpn_out_channels, 1, 1).to(device)

        self.upscale = lambda input: F.interpolate(input, scale_factor=2)
        self.freeze_params()

    def freeze_params(self):
        modules = [
            self.conv3,
            self.conv4,
            self.conv5
        ]
        for module in modules:
            for layer in module:
                if isinstance(layer, nn.Conv2d):
                    for param in layer.parameters():
                        param.requires_grad = False

    def forward(self, x):
        c2 = self.conv2(x)
        c3 = self.conv3(c2)
        c4 = self.conv4(c3)
        c5 = self.conv5(c4)

        p5 = self.lateral_conv5(c5)
        p4 = self.lateral_conv4(c4) + self.upscale(p5)
        p3 = self.lateral_conv3(c3) + self.upscale(p4)

        return [p3, p4, p5]
