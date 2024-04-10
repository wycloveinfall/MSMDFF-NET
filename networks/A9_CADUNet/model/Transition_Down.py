# -*-coding:utf-8-*-

import torch
import torch.nn as nn

class trans_down(nn.Module):
    def __init__(self,in_channels):
        super(trans_down, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv_1x1 = nn.Conv2d(in_channels, in_channels//2,kernel_size=1,stride=1)
        self.avgpool = nn.AvgPool2d(kernel_size=2)

    def forward(self, x):
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.conv_1x1(x)
        x = self.avgpool(x)
        return x

if __name__ == '__main__':
    from torchsummary import summary
    summary(trans_down(96).cuda(),(96,64,64))