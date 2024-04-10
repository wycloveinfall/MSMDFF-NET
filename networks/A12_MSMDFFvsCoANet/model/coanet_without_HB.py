import torch
import torch.nn as nn
from networks.A8_CoANet.model.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d
from networks.A8_CoANet.model.aspp import build_aspp
from networks.A12_MSMDFFvsCoANet.decoder.CoANet_decoder import build_decoder,build_decoder_use_SCMD
from networks.A8_CoANet.model.backbone import build_backbone


class last_conv(nn.Module):
    def __init__(self, num_classes):
        super(last_conv, self).__init__()

        self.seg_branch = nn.Sequential(nn.Conv2d(64, 64, 3, stride=1, padding=1),
                                        nn.ReLU(),
                                        nn.Conv2d(64, num_classes, kernel_size=1, stride=1))
        self._init_weight()

    def forward(self, input):
        seg = self.seg_branch(input)
        return torch.sigmoid(seg)
    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.ConvTranspose2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, SynchronizedBatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class CoANet_base(nn.Module):
    '''2023.11.07 对比实验：
    is_use_SCMD->False: 使用CoANet论文中提出的SCM
    is_use_SCMD->False: 使用本文提出的SCM-D模块'''
    def __init__(self, backbone='resnet', output_stride=16, num_classes=21, num_neighbor=9,
                 sync_bn=True, freeze_bn=False,is_use_SCMD=False):
        super(CoANet_base, self).__init__()
        if sync_bn == True:
            BatchNorm = SynchronizedBatchNorm2d
        else:
            BatchNorm = nn.BatchNorm2d

        self.backbone = build_backbone(backbone, output_stride, BatchNorm)
        self.aspp = build_aspp(backbone, output_stride, BatchNorm)
        ######################## Whether to use SCMD to build decoder for coanet ###################################
        if is_use_SCMD:
            self.decoder = build_decoder_use_SCMD(num_classes, backbone, BatchNorm)
        else:
            self.decoder = build_decoder(num_classes, backbone, BatchNorm)

        # self.connect = build_connect(num_classes, num_neighbor, BatchNorm) #remove hub
        self.last_conv = last_conv(num_classes=num_classes)
        self.freeze_bn = freeze_bn

    def forward(self, input):
        e1, e2, e3, e4 = self.backbone(input)
        e4 = self.aspp(e4)
        x = self.decoder(e1, e2, e3, e4)
        # seg, connect, connect_d1 = self.connect(x) #remove hub
        seg = self.last_conv(x)
        return seg

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, SynchronizedBatchNorm2d):
                m.eval()
            elif isinstance(m, nn.BatchNorm2d):
                m.eval()

    def get_1x_lr_params(self):
        modules = [self.backbone]
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if self.freeze_bn:
                    if isinstance(m[1], nn.Conv2d):
                        for p in m[1].parameters():
                            if p.requires_grad:
                                yield p
                else:
                    if isinstance(m[1], nn.Conv2d) or isinstance(m[1], SynchronizedBatchNorm2d) \
                            or isinstance(m[1], nn.BatchNorm2d):
                        for p in m[1].parameters():
                            if p.requires_grad:
                                yield p

    def get_2x_lr_params(self):
        modules = [self.aspp, self.decoder, self.connect]
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if self.freeze_bn:
                    if isinstance(m[1], nn.Conv2d):
                        for p in m[1].parameters():
                            if p.requires_grad:
                                yield p
                else:
                    if isinstance(m[1], nn.Conv2d) or isinstance(m[1], SynchronizedBatchNorm2d) \
                            or isinstance(m[1], nn.BatchNorm2d):
                        for p in m[1].parameters():
                            if p.requires_grad:
                                yield p

if __name__ == "__main__":
    model = CoANet_base(num_classes=1,
                   backbone='resnet',
                   output_stride=8,
                   sync_bn=False,
                   freeze_bn=False,is_use_SCMD=False)

    from torchsummary import summary

    summary(model.cuda(), (3, 512, 512))
