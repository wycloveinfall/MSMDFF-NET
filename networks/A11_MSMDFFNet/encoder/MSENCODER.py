# -*-coding:utf-8-*-
# !/usr/bin/env python
# @Time    : 2023/5/5 上午9:53
# @Author  : 王玉川 uestc
# @File    : MSENCODER.py
# @Description :
import torch
import torch.nn as nn
from torch.nn import functional as F
from torchvision.transforms import InterpolationMode
from torchvision.transforms.functional import rotate
from functools import partial
tensor_rotate = partial(rotate,interpolation=InterpolationMode.BILINEAR)

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


class connecter(nn.Module):
    # 连接器
    def __init__(self, in_ch, out_ch, scale_factor=0.5):
        super(connecter, self).__init__()
        self.downsample = partial(F.interpolate, scale_factor=scale_factor, mode='area', recompute_scale_factor=True)
        # mode(str)：用于采样的算法。'nearest'| 'linear'| 'bilinear'| 'bicubic'| 'trilinear'| 'area'。默认：'nearest'
        # /home/wyc/software/anconda3/envs/pytroch_test/lib/python3.9/site-packages/torch/nn/functional.py:3502:
        # UserWarning: The default behavior for interpolate/upsample with float scale_factor changed in 1.6.0 to align with other frameworks/libraries,
        # and now uses scale_factor directly, instead of relying on the computed output size. If you wish to restore the old behavior,
        # please set recompute_scale_factor=True. See the documentation of nn.Upsample for details.

        if not in_ch == out_ch:
            shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=1, bias=False),  # 1x1卷积用于增加维度；stride=2用于减半size；为简化不考虑偏差
                nn.BatchNorm2d(out_ch))
        else:
            shortcut = None
        self.connect_conv = ResidualBlock(in_ch, out_ch, shortcut=shortcut)

    def forward(self, x):
        x = self.downsample(x)
        x = self.connect_conv(x)
        return x

class encoder_i(nn.Module):
    def __init__(self, scale_factor, in_c=64, res_layers=64, num_res_blocks=3, stride=1):
        super(encoder_i, self).__init__()
        # 编码器
        self.resnet_i = self.make_layer(in_c, res_layers, num_res_blocks, stride)
        self.connect_i = connecter(3, res_layers, scale_factor=scale_factor)

    def forward(self, x_input, res_input):
        out1 = self.resnet_i(res_input)
        out2 = self.connect_i(x_input)
        out = torch.cat((out1, out2), 1)
        return out

    def make_layer(self, in_ch, out_ch, block_num, stride=1):
        shortcut = None
        # 判断是否使用降采样 维度增加
        if not in_ch == out_ch or not stride == 1:
            shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride, bias=False),  # 1x1卷积用于增加维度；stride=2用于减半size；为简化不考虑偏差
                nn.BatchNorm2d(out_ch))
        layers = []
        layers.append(ResidualBlock(in_ch, out_ch, stride, shortcut))
        for i in range(1, block_num):
            layers.append(ResidualBlock(out_ch, out_ch))  # 后面的几个ResidualBlock,shortcut直接相加
        return nn.Sequential(*layers)