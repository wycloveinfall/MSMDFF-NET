# -*-coding:utf-8-*-
# !/usr/bin/env python
# @Time    : 2023/3/10 上午8:32
# @Author  : 王玉川 uestc
# @File    : CAM.py
# @Description :
import torch
import torch.nn as nn

class multi_scal_residual_50(nn.Module):
    # 实现子module：Residual Block
    def __init__(self, in_ch, out_ch, stride=1, shortcut=None):
        super(multi_scal_residual_50, self).__init__()
        # self.out_ch = out_ch
        self.conv_first = nn.Sequential(
            nn.Conv2d(in_ch, out_ch//4, 1, stride, padding=0, bias=False),
            nn.BatchNorm2d(out_ch//4),
            nn.ReLU(inplace=True))  # inplace = True原地操作,节省显存

        self._1_conv3_3 = nn.Sequential(
            nn.Conv2d(out_ch//16, out_ch//16, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_ch//16),
            nn.ReLU(inplace=True))
        self._2_conv3_3 = nn.Sequential(
            nn.Conv2d(out_ch//16, out_ch//16, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_ch//16),
            nn.ReLU(inplace=True))

        self._3_conv3_3 = nn.Sequential(
            nn.Conv2d(out_ch//16, out_ch//16, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_ch//16),
            nn.ReLU(inplace=True))

        self.conv_last = nn.Sequential(
            nn.Conv2d(out_ch//4, out_ch, 1, stride, padding=0, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True))  # inplace = True原地操作,节省显存
        self.downsample = shortcut

    def forward(self, x):
        N = x.size()[1]//4
        X = self.conv_first(x)

        x2 = X[:,N:2*N,:,:]
        x3 = X[:,2*N:3*N,:,:]
        x4 = X[:,3*N:4*N,:,:]

        y1 = X[:,:N,:,:]
        y2 = self._1_conv3_3(x2)
        y3 = self._2_conv3_3(x3+y2)
        y4 = self._3_conv3_3(x4+y3)
        y = torch.cat([y1,y2,y3,y4],dim=1)
        out = self.conv_last(y)
        residual = x if self.downsample is None else self.downsample(x)
        out += residual
        return out

class Bottleneck(nn.Module):
    expansion: int = 4
    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample = None,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer = None
    ) -> None:
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.))
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = nn.Conv2d(inplanes, width, kernel_size=1, stride=1, bias=False)
        self.bn1 = norm_layer(width)
        ######## multi_scal residual #######################
        self._1_conv3_3 = nn.Sequential(
            nn.Conv2d(width//4, width//4, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(width//4),
            nn.ReLU(inplace=True))
        self._2_conv3_3 = nn.Sequential(
            nn.Conv2d(width//4, width//4, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(width//4),
            nn.ReLU(inplace=True))
        self._3_conv3_3 = nn.Sequential(
            nn.Conv2d(width//4, width//4, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(width//4),
            nn.ReLU(inplace=True))
        ######### end #####################################
        self.conv3 = nn.Conv2d(width, planes * self.expansion, kernel_size=1, stride=stride, bias=False)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        N = out.size()[1] // 4
        x2 = out[:,N:2*N,:,:]
        x3 = out[:,2*N:3*N,:,:]
        x4 = out[:,3*N:4*N,:,:]

        y1 = out[:,:N,:,:]
        y2 = self._1_conv3_3(x2)
        y3 = self._2_conv3_3(x3+y2)
        y4 = self._3_conv3_3(x4+y3)
        y = torch.cat([y1,y2,y3,y4],dim=1)

        out = self.conv3(y)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

