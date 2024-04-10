# -*-coding:utf-8-*-
# !/usr/bin/env python
# @Time    : 2023/2/28 上午8:21
# @Author  : 王玉川 uestc
# @File    : roadcnn.py
# @Description : 根据tf代码复现的roadcnn
# 参考：https://github.com/mitroadmaps/roadtracer/blob/master/roadcnn/model.py
#
'''
[1] F. Bastani et al., “RoadTracer: Automatic Extraction of Road Networks from Aerial Images,” in 2018 IEEE/CVF Conference on Computer Vision and Pattern Recognition, Salt Lake City, UT: IEEE, Jun. 2018, pp. 4720–4728. doi: 10.1109/CVPR.2018.00496.

'''
import torch
import torch.nn as nn
from torchsummary import summary


class _conv_layer(nn.Module):
    def __init__(self, name, input_var, stride, in_ch, out_ch, options={}):
        super(_conv_layer, self).__init__()
        activation = options.get('activation', 'relu')
        dropout = options.get('dropout', None)
        padding = options.get('padding', 1)  # tf代码为same
        batchnorm = options.get('batchnorm', True)
        transpose = options.get('transpose', False)
        if not transpose == True:  # 正常的卷积
            self._convd = nn.Conv2d(in_ch, out_ch, 3, stride, padding=padding, bias=True)
        else:
            self._convd = nn.ConvTranspose2d(in_ch, out_ch, 3, stride=stride,
                                             padding=1, output_padding=padding,
                                             bias=True)
        #
        if batchnorm == True:
            self.bn = nn.BatchNorm2d(out_ch)
        else:
            self.bn = None
        # 随机丢弃层
        if not dropout == None:  # tf.nn.dropout(output, keep_prob=1-dropout) 等价
            self.dropout = torch.nn.Dropout(p=dropout)
        else:
            self.dropout = None

        # 激活函数
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid(inplace=True)
        elif activation == 'none':
            self.activation = None
        else:
            raise Exception('invalid activation {} specified'.format(activation))


    def forward(self, x):
        out = self._convd(x)
        out = out if self.bn is None else self.bn(out)
        out = out if self.dropout is None else self.dropout(out)
        out = out if self.activation is None else self.activation(out)
        return out


class Model(nn.Module):
    def __init__(self, big=False,num_classes=1):
        super(Model, self).__init__()

        self.dropout_factor = 0.3
        #encoder
        self.layer1 = _conv_layer('layer1', 'self.inputs', 2, 3, 128, {'batchnorm': False}) # -> 128x128x128
        self.layer2 = _conv_layer('layer2', 'self.layer1', 1, 128, 128) # -> 128x128x128
        self.layer3 = _conv_layer('layer3', 'self.layer2', 2, 128, 128) # -> 64x64x128
        self.layer4 = _conv_layer('layer4', 'self.layer3', 1, 128, 128) # -> 64x64x128
        self.layer5 = _conv_layer('layer5', 'self.layer4', 2, 128, 256, {'dropout': self.dropout_factor}) # -> 32x32x256
        self.layer6 = _conv_layer('layer6', 'self.layer5', 1, 256, 256, {'dropout': self.dropout_factor}) # -> 32x32x256
        self.layer7 = _conv_layer('layer7', 'self.layer6', 2, 256, 512, {'dropout': self.dropout_factor}) # -> 16x16x512
        self.layer8 = _conv_layer('layer8', 'self.layer7', 2, 512, 512, {'dropout': self.dropout_factor}) # -> 8x8x512
        self.layer9 = _conv_layer('layer9', 'self.layer8', 1, 512, 512, {'dropout': self.dropout_factor}) # -> 8x8x512
        self.layer10 = _conv_layer('layer10', 'self.layer9', 1, 512, 512, {'dropout': self.dropout_factor}) # -> 8x8x512
        self.layer11 = _conv_layer('layer11', 'self.layer10', 1, 512, 512, {'dropout': self.dropout_factor}) # -> 8x8x512
        self.layer12 = _conv_layer('layer12', 'self.layer11', 2, 512, 512, {'transpose': True, 'dropout': self.dropout_factor}) # -> 16x16x512
        #decoder
        self.layer13 = _conv_layer('layer13', 'self.layer13_inputs', 1, 1024, 512, {'dropout': self.dropout_factor}) # -> 16x16x512
        self.layer14 = _conv_layer('layer14', 'self.layer13', 2, 512, 256, {'transpose': True, 'dropout': self.dropout_factor}) # -> 32x32x256

        self.layer15 = _conv_layer('layer15', 'self.layer15_inputs', 1, 512, 256, {'dropout': self.dropout_factor}) # -> 32x32x256
        self.layer16 = _conv_layer('layer16', 'self.layer15', 2, 256, 128, {'transpose': True}) # -> 64x64x128

        self.layer17 = _conv_layer('layer17', 'self.layer17_inputs', 1, 256, 128) # -> 64x64x128
        self.layer18 = _conv_layer('layer18', 'self.layer17', 2, 128, 128, {'transpose': True}) # -> 128x128x128

        self.layer19 = _conv_layer('layer19', 'self.layer19_inputs', 2, 256, 128, {'transpose': True}) # -> 256x256x128
        self.layer20 = _conv_layer('layer20', 'self.layer19', 1, 128, 128) # -> 256x256x128

        self.pre_outputs = _conv_layer('pre_outputs', 'self.layer20', 1, 128, num_classes,
                                       {'activation': 'none', 'batchnorm': False})  # -> 256x256x2
        if num_classes == 1:
            self.last_activation = torch.nn.Sigmoid()
        else:
            self.last_activation = torch.nn.Softmax(dim=1)
    def forward(self, x):
        #encoder
        out = self.layer1(x)
        layer2 = self.layer2(out)

        out = self.layer3(layer2)
        layer4 = self.layer4(out)

        out = self.layer5(layer4)
        layer6 = self.layer6(out)

        layer7 = self.layer7(layer6)
        # center
        out = self.layer8(layer7)
        out = self.layer9(out)
        out = self.layer10(out)
        out = self.layer11(out)
        layer12 = self.layer12(out)
        #decoder
        layer13_inputs = torch.cat([layer7, layer12], axis=1)
        out = self.layer13(layer13_inputs)
        layer14 = self.layer14(out)

        layer15_inputs = torch.cat([layer6, layer14], axis=1)
        out = self.layer15(layer15_inputs)
        layer16 = self.layer16(out)

        layer17_inputs = torch.cat([layer4, layer16], axis=1)
        out = self.layer17(layer17_inputs)
        layer18 = self.layer18(out)

        layer19_inputs = torch.cat([layer2, layer18], axis=1)
        out = self.layer19(layer19_inputs)
        out = self.layer20(out)

        out = self.pre_outputs(out)
        pre_outputs = self.last_activation(out)
        return pre_outputs

if __name__ == '__main__':
    # filters = [64, 128, 256, 512]
    # resnet = models.resnet34(pretrained=False)
    # summary(resnet.cuda(), (3, 512, 512))
    #
    # print('***************\n*****************\n')

    # MY RESNET
    resnet_my = Model()

    # print(self_encoder1)
    summary(resnet_my.cuda(), (3, 256, 256))