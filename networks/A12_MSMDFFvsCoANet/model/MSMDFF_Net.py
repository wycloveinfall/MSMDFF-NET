# -*-coding:utf-8-*-
# !/usr/bin/env python
# @Time    : 2023/2/4 上午10:15
# @Author  : 王玉川 uestc
# @File    : Multi_scale_LinkNet34.py
# @Description :

import torch
import torch.nn as nn
from networks.A12_MSMDFFvsCoANet.InitBlock.multi_D_initblock import initblock_use_SCMD, initblock_use_SCM
from networks.A11_MSMDFFNet.encoder.MSENCODER import encoder_i,connecter,nonlinearity

from networks.A12_MSMDFFvsCoANet.decoder.MSMDFF_decoder import DecoderBlock_Use_SCMD,DecoderBlock_Use_SCM

# https://blog.51cto.com/u_15906550/5921646 旋转特征图
class MSMDFF_Net_base(nn.Module):
    '''2023.11.07 对比实验：
    is_use_SCMD->False: 使用CoANet论文中提出的SCM
    is_use_SCMD->False: 使用本文提出的SCM-D模块'''
    def __init__(self, in_c=3, num_classes=1,init_block_SCMD = True,decoder_use_SCMD=True,):
        super(MSMDFF_Net_base, self).__init__()

        layers = [64, 128, 256, 512]
        if init_block_SCMD:
            self.init_block = initblock_use_SCMD(3,64,stride=2)
        else:
            self.init_block = initblock_use_SCM (3,64,stride=2)

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
        # decoder
        if decoder_use_SCMD:
            self.decoder4 = DecoderBlock_Use_SCMD(layers[3], layers[2],in_p=True)
            self.decoder3 = DecoderBlock_Use_SCMD(layers[2], layers[1],in_p=True)
            self.decoder2 = DecoderBlock_Use_SCMD(layers[1], layers[0],in_p=True)
            self.decoder1 = DecoderBlock_Use_SCMD(layers[0], layers[0],in_p=True)
        else:
            self.decoder4 = DecoderBlock_Use_SCM(layers[3], layers[2],in_p=True)
            self.decoder3 = DecoderBlock_Use_SCM(layers[2], layers[1],in_p=True)
            self.decoder2 = DecoderBlock_Use_SCM(layers[1], layers[0],in_p=True)
            self.decoder1 = DecoderBlock_Use_SCM(layers[0], layers[0],in_p=True)


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
    summary(MSMDFF_Net_base(3,init_block_SCMD=True,decoder_use_SCMD=False),(3,256,256),device='cpu')

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
v2
================================================================
Total params: 24,431,921
Trainable params: 24,431,921
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.75
Forward/backward pass size (MB): 971.12
Params size (MB): 93.20
Estimated Total Size (MB): 1065.08
----------------------------------------------------------------
'''