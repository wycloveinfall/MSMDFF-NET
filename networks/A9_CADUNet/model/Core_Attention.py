# -*-coding:utf-8-*-
import torch
import torch.nn as nn
import torch.nn.functional as F

class core_atention(nn.Module):
    def __init__(self,in_c1,in_c2):
        super(core_atention, self).__init__()

        self.conv_2x2_l = nn.Conv2d(in_c1, in_c1,kernel_size=2,stride=2)
        self.conv_1x1_r = nn.Conv2d(in_c2, in_c1,kernel_size=1,stride=1)

        self.relu = nn.ReLU(inplace=True)
        self.conv_1x1_c = nn.Conv2d(in_c1, in_c1, kernel_size=1, stride=1)
        self.sigmod = nn.Sigmoid()

    def forward(self, x_l, x_r):
        # left
        y_l = self.conv_2x2_l(x_l)
        # do channel and spatial size matching through convolution and up-sampling operations
        # right channel conver
        y_r = self.conv_1x1_r(x_r)
        if not y_l.size()[2]//y_r.size()[2] == 1:#
            y_r = F.interpolate(y_r,
                                scale_factor=y_l.size()[2]//y_r.size()[2],
                                mode='bilinear', align_corners=True)
        # add
        y_c = y_l + y_r
        # center
        y_c = self.relu(y_c)
        y_c = self.conv_1x1_c(y_c)
        y_c = self.sigmod(y_c)
        y_c = F.interpolate(y_c, scale_factor=2, mode='bilinear', align_corners=True)
        #finnal
        out = torch.matmul(x_l,y_c)
        return out

if __name__ == '__main__':
    from torchsummary import summary
    # x_l = torch.rand([2,384,64,64]).cuda()
    # x_r = torch.rand([2,2112,8,8]).cuda()
    x_l = torch.rand([2,768,32,32]).cuda()
    x_r = torch.rand([2,2112,8,8]).cuda()
    model = core_atention(768,2112).cuda()
    model(x_l,x_r)


