# -*-coding:utf-8-*-
# !/usr/bin/env python
# @Time    : 2023/3/10 下午7:59
# @Author  : 王玉川 uestc
# @File    : GAMS_Net.py
# @Description :
'''
GAMS_net在实现的时候，
有两个地方SAM 和 CAM 模块
SAM模块： 1*W*H ->C*W*H
CAM模块： C*1*1 ->C*W*H
使用 repeat 或者 title 都会导致梯度叠加
使用参数 replicated： 来选择使用的 repeat or title
        grad_repeat： True ， 需要除以 一个参数 归一化梯度
                      False 使用梯度叠加的 反向传播
为了验证哪一个最好：
GAMSNet_RT: replicated-repeat grad_repeat-True
GAMSNet_TT: replicated-title grad_repeat-True
GAMSNet_RF: replicated-repeat grad_repeat-False
GAMSNet_TF: replicated-title grad_repeat-False
[1] X. Lu, Y. Zhong, Z. Zheng, and L. Zhang, “GAMSNet: Globally aware road detection network with multi-scale residual learning,” ISPRS Journal of Photogrammetry and Remote Sensing, vol. 175, pp. 340–352, May 2021, doi: 10.1016/j.isprsjprs.2021.03.008.

'''
if __name__ == '__main__':
    __name__ = 'A7_GAMSNet.abc'
import sys,os
sys.path.append(os.path.abspath((os.path.dirname(__file__))))
import torch
import torch.nn as nn
from networks.A7_GAMSNet.MS import Bottleneck
from networks.A7_GAMSNet.GA import GA
from functools import partial
from torch.nn import functional as F
nonlinearity = partial(F.relu, inplace=True)

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


class GAMSNet(nn.Module):
    def __init__(self,block = Bottleneck, num_classes = 1,width_per_group = 64,replicated='tile',grad_repeat=False,norm_layer = None):
        super(GAMSNet, self).__init__()
        layers = [3, 4, 6, 3]
        filters = [256, 512, 1024, 2048]
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.MS_res_block1 = self._make_layer(block,  64, layers[0], stride=2)
        self.MS_res_block2 = self._make_layer(block, 128, layers[1], stride=2)
        self.MS_res_block3 = self._make_layer(block, 256, layers[2], stride=2)
        self.MS_res_block4 = self._make_layer(block, 512, layers[3], stride=2)

        self.GA1 = GA(256,replicated,grad_repeat)
        self.GA2 = GA(512,replicated,grad_repeat)
        self.GA3 = GA(1024,replicated,grad_repeat)

        self.decoder4 = DecoderBlock(filters[3], filters[2])
        self.decoder3 = DecoderBlock(filters[2], filters[1])
        self.decoder2 = DecoderBlock(filters[1], filters[0])
        self.decoder1 = DecoderBlock(filters[0], filters[0])

        self.finaldeconv1 = nn.ConvTranspose2d(filters[0], 32, 3, stride=2, padding=1,output_padding=1)
        self.finalrelu1 = nonlinearity
        self.finalconv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.finalrelu2 = nonlinearity
        self.finaldeconv3 = nn.ConvTranspose2d(32, num_classes, 2, stride=2, padding=0,output_padding=0)

        if num_classes == 1:
           self.last_activation = torch.nn.Sigmoid()
        else:
           self.last_activation = torch.nn.Softmax(dim=1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


    def _make_layer(self, block, planes: int, blocks: int,
                    stride: int = 1):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                norm_layer(planes * block.expansion),)
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        e1 = self.MS_res_block1(x)
        e11 = self.GA1(e1)
        e2 = self.MS_res_block2(e11)
        e22 = self.GA2(e2)
        e3 = self.MS_res_block3(e22)
        e33 = self.GA3(e3)
        e4 = self.MS_res_block4(e33)
        # Decoder
        d4 = self.decoder4(e4) + e3
        d3 = self.decoder3(d4) + e2
        d2 = self.decoder2(d3) + e1
        d1 = self.decoder1(d2)

        out = self.finaldeconv1(d1)
        out = self.finalrelu1(out)
        out = self.finalconv2(out)
        out = self.finalrelu2(out)
        out = self.finaldeconv3(out)

        return self.last_activation(out)

if __name__ == 'A7_GAMSNet.abc':
    from torchsummary import summary
    summary(GAMSNet().cuda(),(3,512,512))