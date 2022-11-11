import torch
from torch import nn
from torch.nn import init
from src import config


class SEAttention(nn.Module):
    def __init__(self, channel=512, reduction=16, feat_h=None, feat_w=None):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )
        self.feat_h = feat_h
        self.feat_w = feat_w

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        if config.onnx_ongoing:
            b = 1
            c = 256
        else:
            b, c, _, _ = x.shape

        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        if self.feat_h is not None and self.feat_w is not None:
            return x * y.repeat(1, 1, self.feat_h, self.feat_w)
        else:
            return x * y.expand_as(x)
