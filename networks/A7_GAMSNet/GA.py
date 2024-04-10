# -*-coding:utf-8-*-
# !/usr/bin/env python
# @Time    : 2023/3/10 上午8:32
# @Author  : 王玉川 uestc
# @File    : SAM.py
# @Description :
import torch.nn
import torch.nn as nn

class GA(nn.Module):
    def __init__(self,in_channel,replicated='tile',grad_repeat=False):
        super(GA, self).__init__()

        self.replicated = replicated
        self.grad_repeat = grad_repeat

        self.SAM_first_conv1_1 = nn.Sequential(
             nn.Conv2d(in_channel, in_channel//16, kernel_size=1, stride=1),
             nn.BatchNorm2d(in_channel//16),
             nn.ReLU(inplace=True))

        self.SAM_atrous1_conv3_3 = nn.Sequential(
             nn.Conv2d(in_channel // 16, in_channel // 16, kernel_size=3,padding=4, stride=1,dilation=4),
             nn.BatchNorm2d(in_channel // 16),
             nn.ReLU(inplace=True))
        self.SAM_atrous2_conv3_3 = nn.Sequential(
             nn.Conv2d(in_channel // 16, in_channel // 16, kernel_size=3,padding=4, stride=1,dilation=4),
             nn.BatchNorm2d(in_channel // 16),
             nn.ReLU(inplace=True))

        self.SAM_last_conv1_1 = nn.Sequential(
             nn.Conv2d(in_channel//16, 1, kernel_size=1, stride=1),
             nn.BatchNorm2d(1),
             nn.ReLU(inplace=True)) # 1*H*W


        self.CAM_Global_Average_pooling = torch.nn.AdaptiveAvgPool2d((1,1))
        self.CAM_FC1 = nn.Linear(in_channel,in_channel//16)
        self.CAM_FC2 = nn.Linear(in_channel//16,in_channel)


    def forward(self, x):
        _,C,H,W = x.size()
        SA_out = self.SAM_first_conv1_1(x)
        SA_out = self.SAM_atrous1_conv3_3(SA_out)
        SA_out = self.SAM_atrous2_conv3_3(SA_out)
        SA_out = self.SAM_last_conv1_1(SA_out)
        if self.grad_repeat == True:
            grad_decay1 = torch.tensor([1/C],dtype=torch.float).cuda()
        else:
            grad_decay1 = torch.tensor([1],dtype=torch.float).cuda()
        if self.replicated == 'tile':
            SA_out_replicated = SA_out.tile(1,C,1,1) * grad_decay1 #Fs
        else:
            SA_out_replicated = SA_out.repeat(1,C,1,1) * grad_decay1 #Fs
        # [repeat or tile]
        CAM_out = self.CAM_Global_Average_pooling(x)
        CAM_out = CAM_out.squeeze()
        CAM_out = self.CAM_FC1(CAM_out)
        CAM_out = self.CAM_FC2(CAM_out)
        CAM_out1 = CAM_out.unsqueeze(dim=-1).unsqueeze(dim=-1)

        if self.grad_repeat == True:
            grad_decay2 = torch.tensor([1/(H*W)],dtype=torch.float).cuda()
        else:
            grad_decay2 = torch.tensor([1],dtype=torch.float).cuda()

        if self.replicated == 'tile':
           CAM_out_replicated = CAM_out1.tile(1, 1, H, W) * grad_decay2 #Fc
        else:
           CAM_out_replicated = CAM_out1.repeat(1, 1, H, W) * grad_decay2  # Fc

        Wg = torch.sigmoid(torch.mul(SA_out_replicated,CAM_out_replicated))
        Og = torch.mul(Wg,x) + x
        return Og


if __name__ == '__main__':
    from torchsummary import summary
    summary(GA(in_channel=64).cuda(),(64,256,256))
