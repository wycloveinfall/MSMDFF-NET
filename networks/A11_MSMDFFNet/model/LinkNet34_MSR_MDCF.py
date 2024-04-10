# -*-coding:utf-8-*-
# !/usr/bin/env python
# @Time    : 2023/2/4 上午10:15
# @Author  : 王玉川 uestc
# @File    : Multi_scale_LinkNet34.py
# @Description :

import torch
import torch.nn as nn
from networks.A11_MSMDFFNet.encoder.MSENCODER import encoder_i,connecter,nonlinearity

from networks.A11_MSMDFFNet.model.decoder import DecoderBlock_v4 as F_DecoderBlock_v4


class LinkNet34_MSR_MDCF(nn.Module):
    # 实现主module:ResNet34
    # use connector add link compare with Multi_scale_LinkNet34_1019v2
    def __init__(self, in_c=3, num_classes=1):
        super(LinkNet34_MSR_MDCF, self).__init__()

        layers = [64, 128, 256, 512]
        self.init_block = nn.Sequential(
            nn.Conv2d(in_c, layers[0], 7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(layers[0]),
            nn.ReLU(inplace=True),
            # nn.MaxPool2d(3, 2, 1)
        )

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
        self.decoder4 = F_DecoderBlock_v4(layers[3], layers[2],in_p=True)
        self.decoder3 = F_DecoderBlock_v4(layers[2], layers[1],in_p=True)
        self.decoder2 = F_DecoderBlock_v4(layers[1], layers[0],in_p=True)
        self.decoder1 = F_DecoderBlock_v4(layers[0], layers[0],in_p=True)

        # self.finaldeconv1 = nn.ConvTranspose2d(layers[0], 32, 4, 2, 1)
        # self.finalrelu1 = nonlinearity
        self.finalconv2 = nn.Conv2d(layers[0], 32, 3, padding=1)
        self.finalrelu2 = nonlinearity
        self.finalconv3 = nn.Conv2d(32, num_classes, 3, padding=1)


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
        d1 = self.decoder1(d2)    # 64*256*256

        # out = self.finaldeconv1(d1)
        # out = self.finalrelu1(out)
        out = self.finalconv2(d1)
        out = self.finalrelu2(out)
        out = self.finalconv3(out)

        # torch.cuda.empty_cache()
        return torch.sigmoid(out)

if __name__ == '__main__':
    from torchsummary import summary
    # summary(MSMDFF_Net(3),(3,256,256),device='cpu')
    summary(LinkNet34_MSR_MDCF(3),(3,256,256),device='cpu')

'''
src
================================================================
Total params: 39,256,881
Trainable params: 39,256,881
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.75
Forward/backward pass size (MB): 986.62
Params size (MB): 149.75
Estimated Total Size (MB): 1137.13
----------------------------------------------------------------
'''