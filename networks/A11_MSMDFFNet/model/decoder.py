import math
import torch
import torch.nn as nn
import torch.nn.functional as F
# from modeling.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d

from torch.nn import functional as F
from torchvision.transforms import InterpolationMode
from torchvision.transforms.functional import rotate
from functools import partial
tensor_rotate = partial(rotate,interpolation=InterpolationMode.BILINEAR)

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

    '''
    stride = 1
    in_channels = 64
    aa = self.deconv2(x.cpu())
    '''

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


class DecoderBlock_v4(nn.Module):
    def __init__(self, in_channels, n_filters, BatchNorm=nn.BatchNorm2d, in_p=True):
        super(DecoderBlock_v4, self).__init__()
        out_pad = 1 if in_p else 0
        stride = 2 if in_p else 1

        self.cbr1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 4, 1),
            BatchNorm(in_channels // 4),
            nn.ReLU(inplace=True), )

        self.cbr2 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 2, 1),
            BatchNorm(in_channels // 2),
            nn.ReLU(inplace=True), )

        self.deconv1 = nn.Conv2d(
            in_channels // 4, in_channels // 4, (1, 9), padding=(0, 4)
        )
        self.deconv2 = nn.Conv2d(
            in_channels // 4, in_channels // 4, (9, 1), padding=(4, 0)
        )
        self.deconv3 = nn.Conv2d(
            in_channels // 4, in_channels // 4, (9, 1), padding=(4, 0)
        )
        self.deconv4 = nn.Conv2d(
            in_channels // 4, in_channels // 4, (1, 9), padding=(0, 4)
        )

        self.cbr3_1 = nn.Sequential(
            nn.Conv2d(in_channels // 4 + in_channels // 2, in_channels // 4, 1),
            BatchNorm(in_channels // 4),
            nn.ReLU(inplace=True), )
        self.cbr3_2 = nn.Sequential(
            nn.Conv2d(in_channels // 4 + in_channels // 2, in_channels // 4, 1),
            BatchNorm(in_channels // 4),
            nn.ReLU(inplace=True), )
        self.cbr3_3 = nn.Sequential(
            nn.Conv2d(in_channels // 4 + in_channels // 2, in_channels // 4, 1),
            BatchNorm(in_channels // 4),
            nn.ReLU(inplace=True), )
        self.cbr3_4 = nn.Sequential(
            nn.Conv2d(in_channels // 4 + in_channels // 2, in_channels // 4, 1),
            BatchNorm(in_channels // 4),
            nn.ReLU(inplace=True), )

        self.deconvbr = nn.Sequential(nn.ConvTranspose2d(in_channels, in_channels // 4 + in_channels // 4,
                                                         3, stride=stride, padding=1, output_padding=out_pad),
                                      nn.BatchNorm2d(in_channels // 4 + in_channels // 4),
                                      nn.ReLU(inplace=True), )

        self.conv3 = nn.Conv2d(in_channels // 4 + in_channels // 4, n_filters, 1)
        self.bn3 = BatchNorm(n_filters)
        self.relu3 = nn.ReLU()

        # self._init_weight()

    def forward(self, x, inp=False):
        x01 = self.cbr1(x)

        x02 = self.cbr2(x)

        x1 = self.deconv1(x01)
        x2 = self.deconv2(x01)
        x3 = self.inv_h_transform(self.deconv3(self.h_transform(x01)))
        x4 = self.inv_v_transform(self.deconv4(self.v_transform(x01)))

        x1 = self.cbr3_1(torch.cat((x1, x02), 1))
        x2 = self.cbr3_2(torch.cat((x2, x02), 1))
        x3 = self.cbr3_3(torch.cat((x3, x02), 1))
        x4 = self.cbr3_4(torch.cat((x4, x02), 1))
        x = torch.cat((x1, x2, x3, x4), 1)

        x = self.deconvbr(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)
        return x

    def _init_weight(self):
        pass
        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         torch.nn.init.kaiming_normal_(m.weight)
        #     elif isinstance(m, nn.ConvTranspose2d):
        #         torch.nn.init.kaiming_normal_(m.weight)
        #     elif isinstance(m, SynchronizedBatchNorm2d):
        #         m.weight.data.fill_(1)
        #         m.bias.data.zero_()
        #     elif isinstance(m, nn.BatchNorm2d):
        #         m.weight.data.fill_(1)
        #         m.bias.data.zero_()

    '''
    x
    N*C*H*W
    当pad只有两个参数时，仅改变最后一个维度,pad=(a,b)表示在最后一个维度左边扩充a列，右边扩充b列
    reshape 是将数据按照行的规则进行重新排列
    '''

    def h_transform(self, x):
        # 获取原始的输入维度
        shape = x.size()  # torch.Size([1, 64, 64, 64])
        # 在最后一个维度左边扩充0列，右边扩充64列
        x = torch.nn.functional.pad(x, (0, shape[-1]))  # torch.Size([1, 64, 64, 128])
        # 消除最后两个维度，
        # x1 = x.reshape(shape[0], shape[1], -1) # torch.Size([1, 64, 8192])
        x = x.reshape(shape[0], shape[1], -1)[..., :-shape[-1]]  # torch.Size([1, 64, 8128])
        x = x.reshape(shape[0], shape[1], shape[2], 2 * shape[3] - 1)  # torch.Size([1, 64,64，127])
        return x

    def inv_h_transform(self, x):
        shape = x.size()  ## torch.Size([1, 32,64，127])
        x = x.reshape(shape[0], shape[1], -1).contiguous()  # torch.Size([1, 32, 8128])
        x = torch.nn.functional.pad(x, (0, shape[-2]))  # torch.Size([1, 32, 8192])
        x = x.reshape(shape[0], shape[1], shape[-2], 2 * shape[-2])  # torch.Size([1, 32, 64, 128])
        x = x[..., 0: shape[-2]]  # torch.Size([1, 32, 64, 64])
        return x

    def v_transform(self, x):
        x = x.permute(0, 1, 3, 2)
        shape = x.size()
        x = torch.nn.functional.pad(x, (0, shape[-1]))
        x = x.reshape(shape[0], shape[1], -1)[..., :-shape[-1]]
        x = x.reshape(shape[0], shape[1], shape[2], 2 * shape[3] - 1)
        return x.permute(0, 1, 3, 2)

    def inv_v_transform(self, x):
        x = x.permute(0, 1, 3, 2)
        shape = x.size()
        x = x.reshape(shape[0], shape[1], -1)
        x = torch.nn.functional.pad(x, (0, shape[-2]))
        x = x.reshape(shape[0], shape[1], shape[-2], 2 * shape[-2])
        x = x[..., 0: shape[-2]]
        return x.permute(0, 1, 3, 2)

class DecoderBlock_v4fix(nn.Module):
    def __init__(self, in_channels, n_filters, BatchNorm=nn.BatchNorm2d, in_p=True,strip=9):
        super(DecoderBlock_v4fix, self).__init__()
        out_pad = 1 if in_p else 0
        stride = 2 if in_p else 1

        self.cbr1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 4, 1),
            BatchNorm(in_channels // 4),
            nn.ReLU(inplace=True), )

        self.cbr2 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 2, 1),
            BatchNorm(in_channels // 2),
            nn.ReLU(inplace=True), )

        self.deconv1 = nn.Conv2d(
            in_channels // 4, in_channels // 4, (1, strip), padding=(0, strip//2)
        )
        self.deconv2 = nn.Conv2d(
            in_channels // 4, in_channels // 4, (strip, 1), padding=(strip//2, 0)
        )
        self.deconv3 = nn.Conv2d(
            in_channels // 4, in_channels // 4, (strip, 1), padding=(strip//2, 0)
        )
        self.deconv4 = nn.Conv2d(
            in_channels // 4, in_channels // 4, (strip, 1), padding=(strip//2, 0)
        )

        self.cbr3_1 = nn.Sequential(
            nn.Conv2d(in_channels // 4 + in_channels // 2, in_channels // 4, 1),
            BatchNorm(in_channels // 4),
            nn.ReLU(inplace=True), )
        self.cbr3_2 = nn.Sequential(
            nn.Conv2d(in_channels // 4 + in_channels // 2, in_channels // 4, 1),
            BatchNorm(in_channels // 4),
            nn.ReLU(inplace=True), )
        self.cbr3_3 = nn.Sequential(
            nn.Conv2d(in_channels // 4 + in_channels // 2, in_channels // 4, 1),
            BatchNorm(in_channels // 4),
            nn.ReLU(inplace=True), )
        self.cbr3_4 = nn.Sequential(
            nn.Conv2d(in_channels // 4 + in_channels // 2, in_channels // 4, 1),
            BatchNorm(in_channels // 4),
            nn.ReLU(inplace=True), )

        self.deconvbr = nn.Sequential(nn.ConvTranspose2d(in_channels, in_channels // 4 + in_channels // 4,
                                                         3, stride=stride, padding=1, output_padding=out_pad),
                                      nn.BatchNorm2d(in_channels // 4 + in_channels // 4),
                                      nn.ReLU(inplace=True), )

        self.conv3 = nn.Conv2d(in_channels // 4 + in_channels // 4, n_filters, 1)
        self.bn3 = BatchNorm(n_filters)
        self.relu3 = nn.ReLU()


    def forward(self, x, inp=False):
        x01 = self.cbr1(x)

        x02 = self.cbr2(x)

        x1 = self.deconv1(x01)
        x2 = self.deconv2(x01)
        x3 = tensor_rotate(self.deconv3(tensor_rotate(x01, 45)), -45)
        x4 = tensor_rotate(self.deconv4(tensor_rotate(x01,135)),-135)

        x1 = self.cbr3_1(torch.cat((x1, x02), 1))
        x2 = self.cbr3_2(torch.cat((x2, x02), 1))
        x3 = self.cbr3_3(torch.cat((x3, x02), 1))
        x4 = self.cbr3_4(torch.cat((x4, x02), 1))
        x = torch.cat((x1, x2, x3, x4), 1)

        x = self.deconvbr(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)
        return x


    '''
    x
    N*C*H*W
    当pad只有两个参数时，仅改变最后一个维度,pad=(a,b)表示在最后一个维度左边扩充a列，右边扩充b列
    reshape 是将数据按照行的规则进行重新排列
    '''




class Decoder(nn.Module):
    def __init__(self, num_classes, backbone, BatchNorm):
        super(Decoder, self).__init__()
        if backbone == 'resnet':
            in_inplanes = 256
        else:
            raise NotImplementedError

        self.decoder4 = DecoderBlock(in_inplanes, 256, BatchNorm)
        self.decoder3 = DecoderBlock(512, 128, BatchNorm)
        self.decoder2 = DecoderBlock(256, 64, BatchNorm, inp=True)
        self.decoder1 = DecoderBlock(128, 64, BatchNorm, inp=True)

        self.conv_e3 = nn.Sequential(nn.Conv2d(1024, 256, 1, bias=False),
                                       BatchNorm(256),
                                       nn.ReLU())

        self.conv_e2 = nn.Sequential(nn.Conv2d(512, 128, 1, bias=False),
                                     BatchNorm(128),
                                     nn.ReLU())

        self.conv_e1 = nn.Sequential(nn.Conv2d(256, 64, 1, bias=False),
                                     BatchNorm(64),
                                     nn.ReLU())

        # self._init_weight()


    def forward(self, e1, e2, e3, e4):
        d4 = torch.cat((self.decoder4(e4), self.conv_e3(e3)), dim=1)
        d3 = torch.cat((self.decoder3(d4), self.conv_e2(e2)), dim=1)
        d2 = torch.cat((self.decoder2(d3), self.conv_e1(e1)), dim=1)
        d1 = self.decoder1(d2)
        x = F.interpolate(d1, scale_factor=2, mode='bilinear', align_corners=True)

        return x

    # def _init_weight(self):
    #     pass
        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         torch.nn.init.kaiming_normal_(m.weight)
        #     elif isinstance(m, SynchronizedBatchNorm2d):
        #         m.weight.data.fill_(1)
        #         m.bias.data.zero_()
        #     elif isinstance(m, nn.BatchNorm2d):
        #         m.weight.data.fill_(1)
        #         m.bias.data.zero_()

def build_decoder(num_classes, backbone, BatchNorm):
    return Decoder(num_classes, backbone, BatchNorm)