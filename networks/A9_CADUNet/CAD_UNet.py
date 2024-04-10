# -*-coding:utf-8-*-
'''

[1] J. Li, Y. Liu, Y. Zhang, and Y. Zhang, “Cascaded Attention DenseUNet (CADUNet) for Road Extraction from Very-High-Resolution Images,” IJGI, vol. 10, no. 5, p. 329, May 2021, doi: 10.3390/ijgi10050329.

'''
import sys,os
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
import torch
import torch.nn as nn

import torch.nn.functional as F

from model.Dense_Block import DenseBlock
from model.Transition_Down import trans_down
from model.Global_Attention import global_atention
from model.Core_Attention import core_atention


class UP(nn.Module):
    def __init__(self,in_channels,out_channels):
        super(UP,self).__init__()
        self.conv1_1x1 = nn.Conv2d(in_channels, out_channels,kernel_size=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2_1x1 = nn.Conv2d(out_channels, out_channels,kernel_size=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.conv1_1x1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        out = F.interpolate(out, scale_factor=2, mode='bilinear', align_corners=True)

        out = self.conv2_1x1(out)
        out = self.bn2(out)
        out = self.relu2(out)

        return out

class Cad_UNet(nn.Module):
    def __init__(self,num_classes=1):
        super(Cad_UNet, self).__init__()
        self.initblock = nn.Sequential(
            nn.Conv2d(3, 96,kernel_size=7,stride=2,padding=3),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True)
        )

        self.maxpooling = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)


        _layer_in = [96,192,384,1056]
        growth_rate = 48
        # encoder
        self.DenseBlock1 = DenseBlock(_layer_in[0],6,growth_rate)
        self.trans_down1 = trans_down(_layer_in[0]+6*growth_rate)

        self.DenseBlock2 = DenseBlock(_layer_in[1],12,growth_rate)
        self.trans_down2 = trans_down(_layer_in[1]+12*growth_rate)

        self.DenseBlock3 = DenseBlock(_layer_in[2],36,growth_rate)
        self.trans_down3 = trans_down(_layer_in[2]+36*growth_rate)

        self.DenseBlock4 = DenseBlock(_layer_in[3],24,growth_rate)
        self.trans_down4 = trans_down(_layer_in[3]+24*growth_rate)
        # center
        self.center_block = nn.Sequential(
            nn.Conv2d(2208,2112,kernel_size=1,stride=1),
            nn.BatchNorm2d(2112),
            nn.ReLU(inplace=True)
        )
        # core attention
        self.CAM1 = core_atention(384,2112)
        self.CAM2 = core_atention(768,2112)
        self.CAM3 = core_atention(2112,2112)
        #
        self.GAM1 = global_atention(96)
        self.GAM2 = global_atention(384)
        self.GAM3 = global_atention(768)
        self.GAM4 = global_atention(2112)

        # up
        self.up1 = UP(384,96)
        self.up2 = UP(768,384)
        self.up3 = UP(2112,768)

        # final
        self.final1 = nn.Sequential(
            nn.Conv2d(96,96,kernel_size=1),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True)
        )
        self.final2 = nn.Sequential(
            nn.Conv2d(192, 96,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True)
        )
        self.final3 = nn.Sequential(
            nn.Conv2d(96, 64,kernel_size=1),
            nn.Dropout(0.3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64,num_classes,kernel_size=1)
        )
        if num_classes == 1:
           self.last_activation = torch.nn.Sigmoid()
        else:
           self.last_activation = torch.nn.Softmax(dim=1)


    def forward(self, x):
        x  = self.initblock(x) #128*128*96
        x1 = self.maxpooling(x) #64*64*96
        # encoder
        e1 = self.DenseBlock1(x1)#64*64*384
        out = self.trans_down1(e1)#32*32*192
        e2 = self.DenseBlock2(out)#32*32*768
        out = self.trans_down2(e2)#16*16*384
        e3 = self.DenseBlock3(out)#16*16*2112
        out = self.trans_down3(e3)#8*8*1056
        e4 = self.DenseBlock4(out)#8*8*2208
        # center
        center = self.center_block(e4)#8*8*2112
        #up1 16*13*2112
        out = F.interpolate(center, scale_factor=2, mode='bilinear', align_corners=True)
        out = self.GAM4(out)#16*16*2112
        # up2
        out = self.up3(out+self.CAM3(e3,center))#32*32*768
        out = self.GAM3(out)#32*32*768
        # up3
        out = self.up2(out + self.CAM2(e2, center))#64*64*348
        out = self.GAM2(out)#64*64*348
        # up4
        out = self.up1(out + self.CAM1(e1, center))#128*128*96
        out = self.GAM1(out)#128*128*96

        # final1
        f1 = self.final1(out+x)#128*128*96
        f2 = self.final2(torch.cat([out,f1],dim=1))#128*128*96
        #256*256*96
        f2 = F.interpolate(f2, scale_factor=2, mode='bilinear', align_corners=True)
        f3 = self.final3(f2)#256*256*1
        out = self.last_activation(f3)
        return out

if __name__ == '__main__':
    from torchsummary import summary
    summary(Cad_UNet(1).cuda(),(3,256,256))