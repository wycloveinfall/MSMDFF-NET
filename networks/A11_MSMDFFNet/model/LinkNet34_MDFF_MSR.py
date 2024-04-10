# -*-coding:utf-8-*-
# !/usr/bin/env python
# @Time    : 2023/2/4 上午10:15
# @Author  : 王玉川 uestc
# @File    : Multi_scale_LinkNet34.py
# @Description :

import torch
import torch.nn as nn
from networks.A11_MSMDFFNet.encoder.MSENCODER import encoder_i,connecter,nonlinearity,ResidualBlock

from networks.A11_MSMDFFNet.model.decoder import DecoderBlock

from networks.A11_MSMDFFNet.init_block.multi_D_initblock import initblock_plus4


class LinkNet34_MDFF_MSR(nn.Module):
    # 实现主module:ResNet34
    # use connector add link compare with Multi_scale_LinkNet34_1019v2
    def __init__(self, in_c=3, num_classes=1):
        super(LinkNet34_MDFF_MSR, self).__init__()

        filters = layers = [64, 128, 256, 512]
        self.init_block = initblock_plus4(3, 64, stride=2)

        # 编码器 512 256 128 64
        self.encoder1 = encoder_i(0.5, in_c=layers[0], res_layers=layers[0], num_res_blocks=3, stride=1)
        # self.max_pool1 = nn.MaxPool2d(3, 2, 1)
        self.encoder2 = encoder_i(0.25, in_c=layers[1], res_layers=layers[1], num_res_blocks=4, stride=2)
        # self.max_pool2 = nn.MaxPool2d(3, 2, 1)
        self.encoder3 = encoder_i(0.125, in_c=layers[2], res_layers=layers[2], num_res_blocks=6, stride=2)
        # self.max_pool3 = nn.MaxPool2d(3, 2, 1)
        self.encoder4 = encoder_i(0.0625, in_c=layers[3], res_layers=layers[3], num_res_blocks=3, stride=2)

        # link
        self.c5 = connecter(1024, 512, scale_factor=1)
        self.c4 = connecter(512, 256, scale_factor=1)
        self.c3 = connecter(256, 128, scale_factor=1)
        self.c2 = connecter(128, 64, scale_factor=1)
        # 解码器
        self.decoder4 = DecoderBlock(filters[3], filters[2])
        self.decoder3 = DecoderBlock(filters[2], filters[1])
        self.decoder2 = DecoderBlock(filters[1], filters[0])
        self.decoder1 = DecoderBlock(filters[0], filters[0])

        # self.finaldeconv1 = nn.ConvTranspose2d(layers[0], 32, 4, 2, 1)
        # self.finalrelu1 = nonlinearity
        self.finalconv2 = nn.Conv2d(layers[0], 32, 3, padding=1)
        self.finalrelu2 = nonlinearity
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
        x1 = self.init_block(x)

        e1 = self.encoder1(x, x1)  # 128*256*256
        e2 = self.encoder2(x, e1)  # 256*128*128
        e3 = self.encoder3(x, e2)  # 512*64*64
        e4 = self.encoder4(x, e3)  # 512*32*32

        c4 = self.c5(e4)

        # Decoder
        d4 = self.decoder4(c4) + self.c4(e3)   # 256*64*64
        d3 = self.decoder3(d4) + self.c3(e2)   # 128*128*128
        d2 = self.decoder2(d3) + self.c2(e1)   # 64*256*256
        out = self.decoder1(d2)

        # out = self.finaldeconv1(out)
        # out = self.finalrelu1(out)
        out = self.finalconv2(out)
        out = self.finalrelu2(out)
        out = self.finalconv3(out)

        # torch.cuda.empty_cache()
        return torch.sigmoid(out)

if __name__ == '__main__':
    from torchsummary import summary
    summary(LinkNet34_MDFF_MSR(3).cuda(),(3,256,256))