# -*-coding:utf-8-*-
import torch
import torch.nn as nn

class Dense_Layer(nn.Module):
    def __init__(self,in_channels,growth_rate=48,F=2):
        super(Dense_Layer, self).__init__()
        self.BN1 = nn.BatchNorm2d(in_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.Conv_1x1 = nn.Conv2d(in_channels,F*growth_rate,kernel_size=1)
        self.BN2 = nn.BatchNorm2d(F*growth_rate)
        self.relu2 = nn.ReLU(inplace=True)
        self.Conv_3x3 = nn.Conv2d(F * growth_rate,growth_rate,kernel_size=3,padding=1)
        self.Dropout = nn.Dropout2d(0.3) # Dropout2d or Dropout

    def forward(self, x):
        x = self.BN1(x)
        x = self.relu1(x)
        x = self.Conv_1x1(x)
        x = self.BN2(x)
        x = self.relu2(x)
        x = self.Conv_3x3(x)
        x = self.Dropout(x)
        return x

class DenseBlock(nn.Module):
    def __init__(self, in_channel, num_layers,growth_rate=48):
        super(DenseBlock, self).__init__()
        _Layer = []
        channel = in_channel
        for i in range(num_layers):
            _Layer.append(Dense_Layer(channel, growth_rate))
            channel += growth_rate
        self.net = nn.Sequential(*_Layer)

    def forward(self, x):
        for layer in self.net:
            out = layer(x)
            x = torch.cat((out, x), dim=1)
        return x


if __name__ == '__main__':
    from torchsummary import summary
    summary(DenseBlock(96,num_layers=6).cuda(),(96,64,64))

