# -*-coding:utf-8-*-
# !/usr/bin/env python
# @Time    : 2023/3/1 下午3:11
# @Author  : 王玉川 uestc
# @File    : LinkNet.py
# @Description :
# https://github.com/snakers4/spacenet-three/blob/master/src/LinkNet.py
'''
A. Chaurasia and E. Culurciello, “LinkNet: Exploiting Encoder Representations for Efficient Semantic Segmentation,” in 2017 IEEE Visual Communications and Image Processing (VCIP), Dec. 2017, pp. 1–4. doi: 10.1109/VCIP.2017.8305148.

'''
import torch.nn as nn
from functools import partial
from torch.nn import functional as F
import torch
from torchvision import models
from torchsummary import summary

'''
LinkNet18
LinkNet34
LinkNet34_src
'''
nonlinearity = partial(F.relu, inplace=True)

class ResidualBlock(nn.Module):
    # 实现子module：Residual Block
    def __init__(self, in_ch, out_ch, stride=1, shortcut=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)

        self.downsample = shortcut

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        residual = x if self.downsample is None else self.downsample(x)
        out += residual
        return F.relu(out)

def BasicBlock(in_ch, out_ch, stride):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 3, stride, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),  # inplace = True原地操作,节省显存
        nn.Conv2d(out_ch, out_ch, 3, stride=1, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, n_filters, stride=2):
        super(DecoderBlock, self).__init__()
        out_pad = 0 if stride == 1 else 1

        self.conv1 = nn.Conv2d(in_channels, in_channels // 4, 1)
        self.norm1 = nn.BatchNorm2d(in_channels // 4)
        self.relu1 = nonlinearity

        self.deconv2 = nn.ConvTranspose2d(in_channels // 4, in_channels // 4, 3, stride=stride, padding=1,
                                          output_padding=out_pad)
        self.norm2 = nn.BatchNorm2d(in_channels // 4)
        self.relu2 = nonlinearity

        self.conv3 = nn.Conv2d(in_channels // 4, n_filters, 1)
        self.norm3 = nn.BatchNorm2d(n_filters)
        self.relu3 = nonlinearity


    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)
        x = self.deconv2(x)
        x = self.norm2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.norm3(x)
        x = self.relu3(x)
        return x

class LinkNet18(nn.Module):
    def __init__(self, num_classes=1, Pretrained = False):
        super(LinkNet18, self).__init__()

        filters = [64, 128, 256, 512]

        resnet = models.resnet18(pretrained=Pretrained)
        self.firstconv = resnet.conv1
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.firstmaxpool = resnet.maxpool
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4

        # self.dblock = Dblock(512) # 该模块 默认加载

        self.decoder4 = DecoderBlock(filters[3], filters[2])
        self.decoder3 = DecoderBlock(filters[2], filters[1])
        self.decoder2 = DecoderBlock(filters[1], filters[0])
        self.decoder1 = DecoderBlock(filters[0], filters[0])

        self.finaldeconv1 = nn.ConvTranspose2d(filters[0], 32, 4, 2, 1)
        self.finalrelu1 = nonlinearity
        self.finalconv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.finalrelu2 = nonlinearity
        self.finalconv3 = nn.Conv2d(32, num_classes, 3, padding=1)

        if num_classes == 1:
           self.last_activation = torch.nn.Sigmoid()
        else:
           self.last_activation = torch.nn.Softmax(dim=1)

    def forward(self, x):
        # Encoder
        x = self.firstconv(x)
        x = self.firstbn(x)
        x = self.firstrelu(x)
        x = self.firstmaxpool(x)
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)

        # Decoder
        d4 = self.decoder4(e4) + e3
        d3 = self.decoder3(d4) + e2
        d2 = self.decoder2(d3) + e1
        d1 = self.decoder1(d2)

        out = self.finaldeconv1(d1)
        out = self.finalrelu1(out)
        out = self.finalconv2(out)
        out = self.finalrelu2(out)
        out = self.finalconv3(out)

        return self.last_activation(out)

class LinkNet34_src(nn.Module):
    # 实现主module:ResNet34 09.15已经校验 和源码 linknet相一致，
    def __init__(self, in_c=3, num_classes=1):
        super(LinkNet34_src, self).__init__()
        # layers = [64,128,256,512]
        # layers = filters = [32, 64, 128, 256]
        layers = filters = [64, 128, 256, 512]
        self.init_block = nn.Sequential(
            nn.Conv2d(in_c, layers[0], 7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(layers[0]),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2, 1)
        )
        # 编码器
        self.encoder1 = self.make_layer(layers[0], layers[0], 3)
        self.encoder2 = self.make_layer(layers[0], layers[1], 4, stride=2)
        self.encoder3 = self.make_layer(layers[1], layers[2], 6, stride=2)
        self.encoder4 = self.make_layer(layers[2], layers[3], 3, stride=2)

        # 连接器
        # 解码器
        self.decoder4 = DecoderBlock(filters[3], filters[2])
        self.decoder3 = DecoderBlock(filters[2], filters[1])
        self.decoder2 = DecoderBlock(filters[1], filters[0])
        self.decoder1 = DecoderBlock(filters[0], filters[0])

        self.finaldeconv1 = nn.ConvTranspose2d(filters[0], 32, 4, 2, 1)
        self.finalrelu1 = nonlinearity
        self.finalconv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.finalrelu2 = nonlinearity
        self.finalconv3 = nn.Conv2d(32, num_classes, 3, padding=1)

        if num_classes == 1:
           self.last_activation = torch.nn.Sigmoid()
        else:
           self.last_activation = torch.nn.Softmax(dim=1)

    def make_layer(self, in_ch, out_ch, block_num, stride=1):
        shortcut = None
        # 判断是否使用降采样 维度增加
        if not in_ch == out_ch:
            shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride, bias=False),  # 1x1卷积用于增加维度；stride=2用于减半size；为简化不考虑偏差
                nn.BatchNorm2d(out_ch))
        layers = []
        layers.append(ResidualBlock(in_ch, out_ch, stride, shortcut))
        for i in range(1, block_num):
            layers.append(ResidualBlock(out_ch, out_ch))  # 后面的几个ResidualBlock,shortcut直接相加
        return nn.Sequential(*layers)


    def forward(self, x):
        x = self.init_block(x)  # B * 64*256*256
        # write_feature_map(x[2], layer_name='x_src')
        e1 = self.encoder1(x)  # B * 64*256*256
        e2 = self.encoder2(e1)  # B * 64*128*128
        e3 = self.encoder3(e2)  # B * 64* 64 *64
        e4 = self.encoder4(e3)  # B * 64* 32* 32

        # Decoder
        d4 = self.decoder4(e4) + e3
        d3 = self.decoder3(d4) + e2
        d2 = self.decoder2(d3) + e1
        d1 = self.decoder1(d2)

        out = self.finaldeconv1(d1)
        out = self.finalrelu1(out)
        out = self.finalconv2(out)
        out = self.finalrelu2(out)
        out = self.finalconv3(out)

        return self.last_activation(out)

class LinkNet34(nn.Module):
    def __init__(self, num_classes=1, Pretrained = False):
        super(LinkNet34, self).__init__()

        filters = [64, 128, 256, 512]

        resnet = models.resnet34(pretrained=Pretrained)
        self.firstconv = resnet.conv1
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.firstmaxpool = resnet.maxpool
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4

        # self.dblock = Dblock(512) # 该模块 默认加载

        self.decoder4 = DecoderBlock(filters[3], filters[2])
        self.decoder3 = DecoderBlock(filters[2], filters[1])
        self.decoder2 = DecoderBlock(filters[1], filters[0])
        self.decoder1 = DecoderBlock(filters[0], filters[0])

        self.finaldeconv1 = nn.ConvTranspose2d(filters[0], 32, 4, 2, 1)
        self.finalrelu1 = nonlinearity
        self.finalconv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.finalrelu2 = nonlinearity
        self.finalconv3 = nn.Conv2d(32, num_classes, 3, padding=1)

        if num_classes == 1:
           self.last_activation = torch.nn.Sigmoid()
        else:
           self.last_activation = torch.nn.Softmax(dim=1)


    def forward(self, x):
        # Encoder
        x = self.firstconv(x)
        x = self.firstbn(x)
        x = self.firstrelu(x)
        x = self.firstmaxpool(x)
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)

        # Decoder
        d4 = self.decoder4(e4) + e3
        d3 = self.decoder3(d4) + e2
        d2 = self.decoder2(d3) + e1
        d1 = self.decoder1(d2)

        out = self.finaldeconv1(d1)
        out = self.finalrelu1(out)
        out = self.finalconv2(out)
        out = self.finalrelu2(out)
        out = self.finalconv3(out)

        return self.last_activation(out)

class LinkNet50(nn.Module):
    def __init__(self, num_classes=1):
        super(LinkNet50, self).__init__()

        filters = [256, 512, 1024, 2048]
        resnet = models.resnet50(pretrained=False)
        self.firstconv = resnet.conv1
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.firstmaxpool = resnet.maxpool
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4

        # self.dblock = Dblock_more_dilate(2048)

        self.decoder4 = DecoderBlock(filters[3], filters[2])
        self.decoder3 = DecoderBlock(filters[2], filters[1])
        self.decoder2 = DecoderBlock(filters[1], filters[0])
        self.decoder1 = DecoderBlock(filters[0], filters[0])

        self.finaldeconv1 = nn.ConvTranspose2d(filters[0], 32, 4, 2, 1)
        self.finalrelu1 = nonlinearity
        self.finalconv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.finalrelu2 = nonlinearity
        self.finalconv3 = nn.Conv2d(32, num_classes, 3, padding=1)

        if num_classes == 1:
           self.last_activation = torch.nn.Sigmoid()
        else:
           self.last_activation = torch.nn.Softmax(dim=1)

    def forward(self, x):
        # Encoder
        x = self.firstconv(x)
        x = self.firstbn(x)
        x = self.firstrelu(x)
        x = self.firstmaxpool(x)
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)

        # Center
        # e4 = self.dblock(e4)

        # Decoder
        d4 = self.decoder4(e4) + e3
        d3 = self.decoder3(d4) + e2
        d2 = self.decoder2(d3) + e1
        d1 = self.decoder1(d2)
        out = self.finaldeconv1(d1)
        out = self.finalrelu1(out)
        out = self.finalconv2(out)
        out = self.finalrelu2(out)
        out = self.finalconv3(out)

        return self.last_activation(out)

class LinkNet101(nn.Module):
    def __init__(self, num_classes=1):
        super(LinkNet101, self).__init__()

        filters = [256, 512, 1024, 2048]
        resnet = models.resnet101(pretrained=False)
        self.firstconv = resnet.conv1
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.firstmaxpool = resnet.maxpool
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4

        # self.dblock = Dblock_more_dilate(2048)

        self.decoder4 = DecoderBlock(filters[3], filters[2])
        self.decoder3 = DecoderBlock(filters[2], filters[1])
        self.decoder2 = DecoderBlock(filters[1], filters[0])
        self.decoder1 = DecoderBlock(filters[0], filters[0])

        self.finaldeconv1 = nn.ConvTranspose2d(filters[0], 32, 4, 2, 1)
        self.finalrelu1 = nonlinearity
        self.finalconv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.finalrelu2 = nonlinearity
        self.finalconv3 = nn.Conv2d(32, num_classes, 3, padding=1)

        if num_classes == 1:
           self.last_activation = torch.nn.Sigmoid()
        else:
           self.last_activation = torch.nn.Softmax(dim=1)

    def forward(self, x):
        # Encoder
        x = self.firstconv(x)
        x = self.firstbn(x)
        x = self.firstrelu(x)
        x = self.firstmaxpool(x)
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)

        # Center
        # e4 = self.dblock(e4)

        # Decoder
        d4 = self.decoder4(e4) + e3
        d3 = self.decoder3(d4) + e2
        d2 = self.decoder2(d3) + e1
        d1 = self.decoder1(d2)
        out = self.finaldeconv1(d1)
        out = self.finalrelu1(out)
        out = self.finalconv2(out)
        out = self.finalrelu2(out)
        out = self.finalconv3(out)

        return self.last_activation(out)


if __name__ == '__main__':
    '''model_list:
        net = DinkNet34_less_pool()
        net = DinkNet34()
        net = DinkNet50()
        net = DinkNet101()
        net = LinkNet34()
        net = ReNetLinkNet34()
        net = BuildLinkNet34()
        net = BuildLinkNet34()

        summary(net.cuda(), (3,512 ,512))

    '''

    net = LinkNet34(num_classes=1)  # Trainable params: 11,548,737
    # net = LinkNet34(num_classes=1) # Trainable params: 21,656,897
    summary(net.cuda(), (3, 512, 512))

'''
256
================================================================
Total params: 21,656,897
Trainable params: 21,656,897
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.75
Forward/backward pass size (MB): 202.62
Params size (MB): 82.61
Estimated Total Size (MB): 285.99
----------------------------------------------------------------
512
================================================================
Total params: 21,656,897
Trainable params: 21,656,897
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 3.00
Forward/backward pass size (MB): 810.50
Params size (MB): 82.61
Estimated Total Size (MB): 896.11
----------------------------------------------------------------
'''