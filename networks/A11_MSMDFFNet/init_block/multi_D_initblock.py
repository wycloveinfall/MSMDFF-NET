#-*-coding:utf-8-*-

import torch.nn as nn
import torch
from torchvision.transforms import InterpolationMode
from torchvision.transforms.functional import rotate
from functools import partial
from torch.nn import functional as F

nonlinearity = partial(F.relu,inplace=True)
tensor_rotate = partial(rotate,interpolation=InterpolationMode.BILINEAR)


class filters_conv(nn.Module):
    # 激活函数用的relu bing xing
    def __init__(self, in_ch, out_ch, kernel_size=3,stride=1,padding=1,dilation=1,bias = False):
        super(filters_conv, self).__init__()

        self.filter_i = nn.Sequential(nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding=padding,dilation=dilation, bias=bias),
                                      nn.BatchNorm2d(out_ch),
                                      nn.ELU(inplace=True))
    def forward(self, x):
        out = self.filter_i(x)
        return out

class initblock_plus4(nn.Module):
    # 激活函数用的relu bing xing
    def __init__(self, in_ch, out_ch, kernel_size=3,stride=1,padding=1,dilation=1,bias = False,strip=9):
        super(initblock_plus4, self).__init__()



        self.conv_a0 = nn.Sequential(nn.Conv2d(in_ch, 32, kernel_size, stride, padding=padding,dilation=dilation, bias=bias),
                                      nn.BatchNorm2d(32),
                                      nn.ELU(inplace=True)
                                      )


        self.multi_conv1 = nn.Conv2d(in_ch, 8, (1, strip), stride=stride,padding=(0, strip//2))
        self.multi_conv2 = nn.Conv2d(in_ch, 8, (strip, 1), stride=stride,padding=(strip//2, 0))
        self.multi_conv3 = nn.Conv2d(in_ch, 8, (1, strip), stride=stride,padding=(0, strip//2))
        self.multi_conv4 = nn.Conv2d(in_ch, 8, (1, strip), stride=stride,padding=(0, strip//2))

        self.channel_concern = nn.Sequential(nn.Conv2d(32, 32, 1, 1, padding=0,dilation=dilation, bias=bias),
                                      nn.BatchNorm2d(32),
                                      nn.ELU(inplace=True)
                                      )

        self.angle = [0,45,90,135,180]
    def forward(self, x):


        x1 = self.multi_conv1(x)
        x2 = self.multi_conv2(x)
        x3 = self.conv_a0(x)
        x4 = self.multi_conv3(tensor_rotate(x,self.angle[1]))
        x5 = self.multi_conv4(tensor_rotate(x,self.angle[3]))

        out = torch.cat((x1,
                         x2,
                         tensor_rotate(x4,-self.angle[1]),
                         tensor_rotate(x5,-self.angle[3]),
                         ), 1)
        out = torch.cat((self.channel_concern(out),
                         x3),1)
        return out

class initblock_plus4_use_SCM(nn.Module):
    # 激活函数用的relu bing xing
    def __init__(self, in_ch, out_ch, kernel_size=3,stride=1,padding=1,dilation=1,bias = False,strip=9):
        super(initblock_plus4_use_SCM, self).__init__()

        # self.filter_R = filters_conv(1,3)
        # self.filter_G = filters_conv(1,3)
        # self.filter_B = filters_conv(1,3)

        self.conv_a0 = nn.Sequential(nn.Conv2d(in_ch, 32, kernel_size, stride, padding=padding,dilation=dilation, bias=bias),
                                      nn.BatchNorm2d(32),
                                      nn.ELU(inplace=True)
                                      )


        self.multi_conv1 = nn.Conv2d(in_ch, 8, (1, strip), stride=stride,padding=(0, strip//2))
        self.multi_conv2 = nn.Conv2d(in_ch, 8, (strip, 1), stride=stride,padding=(strip//2, 0))
        self.multi_conv3 = nn.Conv2d(in_ch, 8, (1, strip), stride=stride,padding=(0, strip//2-1)) #这里为了使用 SCM模块 与原来的代码相比-1
        self.multi_conv4 = nn.Conv2d(in_ch, 8, (strip, 1), stride=stride,padding=(strip//2-1, 0))

        self.channel_concern = nn.Sequential(nn.Conv2d(32, 32, 1, 1, padding=0,dilation=dilation, bias=bias),
                                      nn.BatchNorm2d(32),
                                      nn.ELU(inplace=True)
                                      )

        self.angle = [0,45,90,135,180]
    def forward(self, x):
        # R = self.filter_R(x[:,0,:,:].unsqueeze(1))
        # G = self.filter_G(x[:,1,:,:].unsqueeze(1))
        # B = self.filter_B(x[:,2,:,:].unsqueeze(1))
        # x = torch.cat((R,
        #                G,
        #                B), 1)

        x1 = self.multi_conv1(x)
        x2 = self.multi_conv2(x)
        x3 = self.conv_a0(x)
        x4 = self.multi_conv3(self.h_transform(x))
        x5 = self.multi_conv4(self.v_transform(x))

        out = torch.cat((x1,
                         x2,
                         self.inv_h_transform(x4),
                         self.inv_v_transform(x5),
                         ), 1)
        out = torch.cat((self.channel_concern(out),
                         x3),1)
        return out


    def h_transform(self, x):
        shape = x.size() # 2 3 512 512
        x = torch.nn.functional.pad(x, (0, shape[-1])) # 2 3 512 1024
        x = x.reshape(shape[0], shape[1], -1)[..., :-shape[-1]] # 2 64 64 128->2 64 8192->2 64 8128(-64)
        x = x.reshape(shape[0], shape[1], shape[2], 2*shape[3]-1)
        return x
    '''
    x1 = x.reshape(shape[0], shape[1], -1)
    x2 = x1[..., :-shape[-1]]
    x3 = x2.reshape(shape[0], shape[1], shape[2], 2*shape[3]-1)
    '''
    def inv_h_transform(self, x):
        shape = x.size() # 2*8*256*512
        x = x.reshape(shape[0], shape[1], -1).contiguous() # 2*8*131072
        x = torch.nn.functional.pad(x, (0, shape[-2])) # 22*8*131072(+256)
        x = x.reshape(shape[0], shape[1], shape[-2], 2*shape[-2]) # 2*32*64*128
        x = x[..., 0: shape[-2]]
        return x

    def v_transform(self, x):
        x = x.permute(0, 1, 3, 2)
        shape = x.size()
        x = torch.nn.functional.pad(x, (0, shape[-1]))
        x = x.reshape(shape[0], shape[1], -1)[..., :-shape[-1]]
        x = x.reshape(shape[0], shape[1], shape[2], 2*shape[3]-1)
        return x.permute(0, 1, 3, 2)

    def inv_v_transform(self, x):
        x = x.permute(0, 1, 3, 2)
        shape = x.size()
        x = x.reshape(shape[0], shape[1], -1)
        x = torch.nn.functional.pad(x, (0, shape[-2]))
        x = x.reshape(shape[0], shape[1], shape[-2], 2*shape[-2])
        x = x[..., 0: shape[-2]]
        return x.permute(0, 1, 3, 2)


if __name__ == '__main__':
    from torchsummary import summary
