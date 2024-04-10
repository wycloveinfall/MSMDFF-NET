# -*-coding:utf-8-*-
# !/usr/bin/env python
# @Time    : 2023/2/4 上午10:15
# @Author  : 王玉川 uestc
# @File    : Multi_scale_LinkNet34.py
# @Description :

import torch
import torch.nn as nn
from networks.A11_MSMDFFNet.encoder.MSENCODER import nonlinearity,ResidualBlock
from networks.A11_MSMDFFNet.model.decoder import DecoderBlock_v4fix as F_DecoderBlock_v4fix

class LinkNet34_MDCF_s2(nn.Module):
    def __init__(self, in_c=3, num_classes=1):
        super(LinkNet34_MDCF_s2, self).__init__()
        layers = [64, 128, 256, 512]
        self.init_block = nn.Sequential(
            nn.Conv2d(in_c, layers[0], 7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(layers[0]),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2, 1)
        )

        # 编码器 512 256 128 64
        self.encoder1 = self.make_layer(layers[0], layers[0], 3)
        self.encoder2 = self.make_layer(layers[0], layers[1], 4, stride=2)
        self.encoder3 = self.make_layer(layers[1], layers[2], 6, stride=2)
        self.encoder4 = self.make_layer(layers[2], layers[3], 3, stride=2)
        # 解码器
        self.decoder4 = F_DecoderBlock_v4fix(layers[3], layers[2],in_p=True)
        self.decoder3 = F_DecoderBlock_v4fix(layers[2], layers[1],in_p=True)
        self.decoder2 = F_DecoderBlock_v4fix(layers[1], layers[0],in_p=True)
        self.decoder1 = F_DecoderBlock_v4fix(layers[0], layers[0],in_p=True)

        self.finaldeconv1 = nn.ConvTranspose2d(layers[0], 32, 4, 2, 1)
        self.finalrelu1 = nonlinearity
        # self.finalconv2 = nn.Conv2d(32, 32, 3, padding=1)
        # self.finalrelu2 = nonlinearity
        self.finalconv3 = nn.Conv2d(32, num_classes, 3, padding=1)

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
        x = self.init_block(x)

        e1 = self.encoder1(x)  # B * 64*256*256
        e2 = self.encoder2(e1)  # B * 64*128*128
        e3 = self.encoder3(e2)  # B * 64* 64 *64
        e4 = self.encoder4(e3)  # B * 64* 32* 32

        # Decoder
        d4 = self.decoder4(e4) + e3
        d3 = self.decoder3(d4) + e2
        d2 = self.decoder2(d3) + e1
        out = self.decoder1(d2)

        out = self.finaldeconv1(out)
        out = self.finalrelu1(out)
        # out = self.finalconv2(out)
        # out = self.finalrelu2(out)
        out = self.finalconv3(out)

        return torch.sigmoid(out)

class LinkNet34_MDCF_s2_strip_size_3(nn.Module):
    def __init__(self, in_c=3, num_classes=1):
        super(LinkNet34_MDCF_s2_strip_size_3, self).__init__()
        layers = [64, 128, 256, 512]
        self.init_block = nn.Sequential(
            nn.Conv2d(in_c, layers[0], 7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(layers[0]),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2, 1)
        )

        # 编码器 512 256 128 64
        self.encoder1 = self.make_layer(layers[0], layers[0], 3)
        self.encoder2 = self.make_layer(layers[0], layers[1], 4, stride=2)
        self.encoder3 = self.make_layer(layers[1], layers[2], 6, stride=2)
        self.encoder4 = self.make_layer(layers[2], layers[3], 3, stride=2)
        # 解码器
        self.decoder4 = F_DecoderBlock_v4fix(layers[3], layers[2],in_p=True)
        self.decoder3 = F_DecoderBlock_v4fix(layers[2], layers[1],in_p=True)
        self.decoder2 = F_DecoderBlock_v4fix(layers[1], layers[0],in_p=True)
        self.decoder1 = F_DecoderBlock_v4fix(layers[0], layers[0],in_p=True)

        self.finaldeconv1 = nn.ConvTranspose2d(layers[0], 32, 4, 2, 1)
        self.finalrelu1 = nonlinearity
        # self.finalconv2 = nn.Conv2d(32, 32, 3, padding=1)
        # self.finalrelu2 = nonlinearity
        self.finalconv3 = nn.Conv2d(32, num_classes, 3, padding=1)

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
        x = self.init_block(x)

        e1 = self.encoder1(x)  # B * 64*256*256
        e2 = self.encoder2(e1)  # B * 64*128*128
        e3 = self.encoder3(e2)  # B * 64* 64 *64
        e4 = self.encoder4(e3)  # B * 64* 32* 32

        # Decoder
        d4 = self.decoder4(e4) + e3
        d3 = self.decoder3(d4) + e2
        d2 = self.decoder2(d3) + e1
        out = self.decoder1(d2)

        out = self.finaldeconv1(out)
        out = self.finalrelu1(out)
        # out = self.finalconv2(out)
        # out = self.finalrelu2(out)
        out = self.finalconv3(out)

        return torch.sigmoid(out)
class LinkNet34_MDCF_s2_strip_size_5(nn.Module):
    def __init__(self, in_c=3, num_classes=1):
        super(LinkNet34_MDCF_s2_strip_size_5, self).__init__()
        layers = [64, 128, 256, 512]
        self.init_block = nn.Sequential(
            nn.Conv2d(in_c, layers[0], 7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(layers[0]),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2, 1)
        )

        # 编码器 512 256 128 64
        self.encoder1 = self.make_layer(layers[0], layers[0], 3)
        self.encoder2 = self.make_layer(layers[0], layers[1], 4, stride=2)
        self.encoder3 = self.make_layer(layers[1], layers[2], 6, stride=2)
        self.encoder4 = self.make_layer(layers[2], layers[3], 3, stride=2)
        # 解码器
        self.decoder4 = F_DecoderBlock_v4fix(layers[3], layers[2],in_p=True)
        self.decoder3 = F_DecoderBlock_v4fix(layers[2], layers[1],in_p=True)
        self.decoder2 = F_DecoderBlock_v4fix(layers[1], layers[0],in_p=True)
        self.decoder1 = F_DecoderBlock_v4fix(layers[0], layers[0],in_p=True)

        self.finaldeconv1 = nn.ConvTranspose2d(layers[0], 32, 4, 2, 1)
        self.finalrelu1 = nonlinearity
        # self.finalconv2 = nn.Conv2d(32, 32, 3, padding=1)
        # self.finalrelu2 = nonlinearity
        self.finalconv3 = nn.Conv2d(32, num_classes, 3, padding=1)

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
        x = self.init_block(x)

        e1 = self.encoder1(x)  # B * 64*256*256
        e2 = self.encoder2(e1)  # B * 64*128*128
        e3 = self.encoder3(e2)  # B * 64* 64 *64
        e4 = self.encoder4(e3)  # B * 64* 32* 32

        # Decoder
        d4 = self.decoder4(e4) + e3
        d3 = self.decoder3(d4) + e2
        d2 = self.decoder2(d3) + e1
        out = self.decoder1(d2)

        out = self.finaldeconv1(out)
        out = self.finalrelu1(out)
        # out = self.finalconv2(out)
        # out = self.finalrelu2(out)
        out = self.finalconv3(out)

        return torch.sigmoid(out)
class LinkNet34_MDCF_s2_strip_size_7(nn.Module):
    def __init__(self, in_c=3, num_classes=1):
        super(LinkNet34_MDCF_s2_strip_size_7, self).__init__()
        layers = [64, 128, 256, 512]
        self.init_block = nn.Sequential(
            nn.Conv2d(in_c, layers[0], 7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(layers[0]),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2, 1)
        )

        # 编码器 512 256 128 64
        self.encoder1 = self.make_layer(layers[0], layers[0], 3)
        self.encoder2 = self.make_layer(layers[0], layers[1], 4, stride=2)
        self.encoder3 = self.make_layer(layers[1], layers[2], 6, stride=2)
        self.encoder4 = self.make_layer(layers[2], layers[3], 3, stride=2)
        # 解码器
        self.decoder4 = F_DecoderBlock_v4fix(layers[3], layers[2],in_p=True)
        self.decoder3 = F_DecoderBlock_v4fix(layers[2], layers[1],in_p=True)
        self.decoder2 = F_DecoderBlock_v4fix(layers[1], layers[0],in_p=True)
        self.decoder1 = F_DecoderBlock_v4fix(layers[0], layers[0],in_p=True)

        self.finaldeconv1 = nn.ConvTranspose2d(layers[0], 32, 4, 2, 1)
        self.finalrelu1 = nonlinearity
        # self.finalconv2 = nn.Conv2d(32, 32, 3, padding=1)
        # self.finalrelu2 = nonlinearity
        self.finalconv3 = nn.Conv2d(32, num_classes, 3, padding=1)

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
        x = self.init_block(x)

        e1 = self.encoder1(x)  # B * 64*256*256
        e2 = self.encoder2(e1)  # B * 64*128*128
        e3 = self.encoder3(e2)  # B * 64* 64 *64
        e4 = self.encoder4(e3)  # B * 64* 32* 32

        # Decoder
        d4 = self.decoder4(e4) + e3
        d3 = self.decoder3(d4) + e2
        d2 = self.decoder2(d3) + e1
        out = self.decoder1(d2)

        out = self.finaldeconv1(out)
        out = self.finalrelu1(out)
        # out = self.finalconv2(out)
        # out = self.finalrelu2(out)
        out = self.finalconv3(out)

        return torch.sigmoid(out)
class LinkNet34_MDCF_s2_strip_size_9(nn.Module):
    def __init__(self, in_c=3, num_classes=1):
        super(LinkNet34_MDCF_s2_strip_size_9, self).__init__()
        layers = [64, 128, 256, 512]
        self.init_block = nn.Sequential(
            nn.Conv2d(in_c, layers[0], 7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(layers[0]),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2, 1)
        )

        # 编码器 512 256 128 64
        self.encoder1 = self.make_layer(layers[0], layers[0], 3)
        self.encoder2 = self.make_layer(layers[0], layers[1], 4, stride=2)
        self.encoder3 = self.make_layer(layers[1], layers[2], 6, stride=2)
        self.encoder4 = self.make_layer(layers[2], layers[3], 3, stride=2)
        # 解码器
        self.decoder4 = F_DecoderBlock_v4fix(layers[3], layers[2],in_p=True)
        self.decoder3 = F_DecoderBlock_v4fix(layers[2], layers[1],in_p=True)
        self.decoder2 = F_DecoderBlock_v4fix(layers[1], layers[0],in_p=True)
        self.decoder1 = F_DecoderBlock_v4fix(layers[0], layers[0],in_p=True)

        self.finaldeconv1 = nn.ConvTranspose2d(layers[0], 32, 4, 2, 1)
        self.finalrelu1 = nonlinearity
        # self.finalconv2 = nn.Conv2d(32, 32, 3, padding=1)
        # self.finalrelu2 = nonlinearity
        self.finalconv3 = nn.Conv2d(32, num_classes, 3, padding=1)

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
        x = self.init_block(x)

        e1 = self.encoder1(x)  # B * 64*256*256
        e2 = self.encoder2(e1)  # B * 64*128*128
        e3 = self.encoder3(e2)  # B * 64* 64 *64
        e4 = self.encoder4(e3)  # B * 64* 32* 32

        # Decoder
        d4 = self.decoder4(e4) + e3
        d3 = self.decoder3(d4) + e2
        d2 = self.decoder2(d3) + e1
        out = self.decoder1(d2)

        out = self.finaldeconv1(out)
        out = self.finalrelu1(out)
        # out = self.finalconv2(out)
        # out = self.finalrelu2(out)
        out = self.finalconv3(out)

        return torch.sigmoid(out)
class LinkNet34_MDCF_s2_strip_size_11(nn.Module):
    def __init__(self, in_c=3, num_classes=1):
        super(LinkNet34_MDCF_s2_strip_size_11, self).__init__()
        layers = [64, 128, 256, 512]
        self.init_block = nn.Sequential(
            nn.Conv2d(in_c, layers[0], 7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(layers[0]),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2, 1)
        )

        # 编码器 512 256 128 64
        self.encoder1 = self.make_layer(layers[0], layers[0], 3)
        self.encoder2 = self.make_layer(layers[0], layers[1], 4, stride=2)
        self.encoder3 = self.make_layer(layers[1], layers[2], 6, stride=2)
        self.encoder4 = self.make_layer(layers[2], layers[3], 3, stride=2)
        # 解码器
        self.decoder4 = F_DecoderBlock_v4fix(layers[3], layers[2],in_p=True)
        self.decoder3 = F_DecoderBlock_v4fix(layers[2], layers[1],in_p=True)
        self.decoder2 = F_DecoderBlock_v4fix(layers[1], layers[0],in_p=True)
        self.decoder1 = F_DecoderBlock_v4fix(layers[0], layers[0],in_p=True)

        self.finaldeconv1 = nn.ConvTranspose2d(layers[0], 32, 4, 2, 1)
        self.finalrelu1 = nonlinearity
        # self.finalconv2 = nn.Conv2d(32, 32, 3, padding=1)
        # self.finalrelu2 = nonlinearity
        self.finalconv3 = nn.Conv2d(32, num_classes, 3, padding=1)

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
        x = self.init_block(x)

        e1 = self.encoder1(x)  # B * 64*256*256
        e2 = self.encoder2(e1)  # B * 64*128*128
        e3 = self.encoder3(e2)  # B * 64* 64 *64
        e4 = self.encoder4(e3)  # B * 64* 32* 32

        # Decoder
        d4 = self.decoder4(e4) + e3
        d3 = self.decoder3(d4) + e2
        d2 = self.decoder2(d3) + e1
        out = self.decoder1(d2)

        out = self.finaldeconv1(out)
        out = self.finalrelu1(out)
        # out = self.finalconv2(out)
        # out = self.finalrelu2(out)
        out = self.finalconv3(out)

        return torch.sigmoid(out)
class LinkNet34_MDCF_s2_strip_size_13(nn.Module):
    def __init__(self, in_c=3, num_classes=1):
        super(LinkNet34_MDCF_s2_strip_size_13, self).__init__()
        layers = [64, 128, 256, 512]
        self.init_block = nn.Sequential(
            nn.Conv2d(in_c, layers[0], 7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(layers[0]),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2, 1)
        )

        # 编码器 512 256 128 64
        self.encoder1 = self.make_layer(layers[0], layers[0], 3)
        self.encoder2 = self.make_layer(layers[0], layers[1], 4, stride=2)
        self.encoder3 = self.make_layer(layers[1], layers[2], 6, stride=2)
        self.encoder4 = self.make_layer(layers[2], layers[3], 3, stride=2)
        # 解码器
        self.decoder4 = F_DecoderBlock_v4fix(layers[3], layers[2],in_p=True)
        self.decoder3 = F_DecoderBlock_v4fix(layers[2], layers[1],in_p=True)
        self.decoder2 = F_DecoderBlock_v4fix(layers[1], layers[0],in_p=True)
        self.decoder1 = F_DecoderBlock_v4fix(layers[0], layers[0],in_p=True)

        self.finaldeconv1 = nn.ConvTranspose2d(layers[0], 32, 4, 2, 1)
        self.finalrelu1 = nonlinearity
        # self.finalconv2 = nn.Conv2d(32, 32, 3, padding=1)
        # self.finalrelu2 = nonlinearity
        self.finalconv3 = nn.Conv2d(32, num_classes, 3, padding=1)

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
        x = self.init_block(x)

        e1 = self.encoder1(x)  # B * 64*256*256
        e2 = self.encoder2(e1)  # B * 64*128*128
        e3 = self.encoder3(e2)  # B * 64* 64 *64
        e4 = self.encoder4(e3)  # B * 64* 32* 32

        # Decoder
        d4 = self.decoder4(e4) + e3
        d3 = self.decoder3(d4) + e2
        d2 = self.decoder2(d3) + e1
        out = self.decoder1(d2)

        out = self.finaldeconv1(out)
        out = self.finalrelu1(out)
        # out = self.finalconv2(out)
        # out = self.finalrelu2(out)
        out = self.finalconv3(out)

        return torch.sigmoid(out)
if __name__ == '__main__':
    from torchsummary import summary
    # summary(LinkNet34_MDCF_s(3).cuda(),(3,256,256))
    # summary(LinkNet34_MDCF(3).cuda(),(3,256,256))
    # summary(LinkNet34_MDCF_s1(3).cuda(),(3,256,256))
    summary(LinkNet34_MDCF_s2_strip_size_13(3).cuda(),(3,256,256))