# -*-coding:utf-8-*-
# !/usr/bin/env python
# @Time    : 2023/2/22
# @Author  : wangyuchuan uestc
# @File    : model_segment.py
# @Description : tensorflow->pytorch
# 源文件：https://github.com/mitroadmaps/roadtracer/tree/master/deeproadmapper
# paper
'''
G. Mattyus, W. Luo, and R. Urtasun, “DeepRoadMapper: Extracting Road Topology from Aerial Images,” in 2017 IEEE International Conference on Computer Vision (ICCV), Venice: IEEE, Oct. 2017, pp. 3458–3466. doi: 10.1109/ICCV.2017.372.
'''
import torch
import torch.nn as nn
from torchsummary import summary

class bn_relu_conv_layer(nn.Module):
    def __init__(self,name, input_var, stride, in_channels, out_channels, options = {}):
        super(bn_relu_conv_layer, self).__init__()
        padding = options.get('padding', 1)
        transpose = options.get('transpose', False)
        first_block = options.get('first_block', False)

        if not first_block == True:
            self.bn1 = nn.BatchNorm2d(in_channels)
            self.relu1 = nn.ReLU(inplace=True)
        else:
            self.bn1 = None
            self.relu1 = None

        if not transpose == True:#正常的卷积
            self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, padding=padding, bias=True)
        else:
            self.conv1 = nn.ConvTranspose2d(in_channels, out_channels, 3, stride=stride,
                                             padding=1,output_padding=padding,
                                             bias=True)
    def forward(self, x):
        out = x if self.bn1 is None else self.bn1(x)
        out = out if self.relu1 is None else self.relu1(out)
        out = self.conv1(out)
        return out

class _residual_layer(nn.Module):
    def __init__(self,name, input_var, stride, in_channels, out_channels, options = {}):
        super(_residual_layer, self).__init__()
        padding = options.get('padding', 1)
        transpose = options.get('transpose', False)
        first_block = options.get('first_block', False)

        self.name = name
        self.stride = stride
        self.transpose = transpose

        self.conv1 = bn_relu_conv_layer('conv1', input_var, stride, in_channels, out_channels,
                                   {'padding': padding, 'transpose': transpose, 'first_block': first_block})
        self.conv2 = bn_relu_conv_layer('conv2', input_var, 1, out_channels, out_channels, {'padding': padding})

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        if self.stride == 1:
            out = out + x
        # 源码中没有用到 not stride == 1 的情况
        else:
            print('_residual_layer error in %s' % self.name)
            exit(0)
        return out

class _conv_layer(nn.Module):
    def __init__(self,name,input_var,stride, in_ch, out_ch,options={}):
        super(_conv_layer, self).__init__()
        activation = options.get('activation', 'relu')
        dropout = options.get('dropout', None)
        padding = options.get('padding', 1) # tf代码为same
        batchnorm = options.get('batchnorm', True)
        transpose = options.get('transpose', False)
        if not transpose == True:#正常的卷积
            self._convd = nn.Conv2d(in_ch, out_ch, 3, stride, padding=padding, bias=True)
        else:
            self._convd = nn.ConvTranspose2d(in_ch, out_ch, 3, stride=stride,
                                             padding=1,output_padding=padding,
                                             bias=True)
        #
        if batchnorm == True:
           self.bn = nn.BatchNorm2d(out_ch)
        else:
           self.bn = None
        #随机丢弃层
        if not dropout == None:# tf.nn.dropout(output, keep_prob=1-dropout) 等价
           self.dropout = torch.nn.Dropout(p=dropout)
        else:
           self.dropout = None
        #激活函数
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
    def __init__(self, mode=0, big=False,num_classes=1):
        super(Model, self).__init__()
        self.layer1 = _conv_layer('layer1', 'self.inputs', 1, 3, 16, {'batchnorm': False}) # -> 1024x1024
        self.layer2 = _residual_layer('layer2', 'self.layer1', 1, 16, 16, {'first_block': True}) # -> 1024x1024
        self.layer3 = _residual_layer('layer3', 'self.layer2', 1, 16, 16) # -> 1024x1024
        self.layer4 = _residual_layer('layer4', 'self.layer3', 1, 16, 16) # -> 1024x1024
        self.layer5 = _residual_layer('layer5', 'self.layer4', 1, 16, 16) # -> 1024x1024
        self.layer6 = _residual_layer('layer6', 'self.layer5', 1, 16, 16) # -> 1024x1024
        self.layer7 = _residual_layer('layer7', 'self.layer6', 1, 16, 16) # -> 1024x1024

        self.layer8 = _conv_layer('layer8', 'self.layer7', 2, 16, 32) # -> 512x512
        self.layer9 = _residual_layer('layer9', 'self.layer8', 1, 32, 32) # -> 512x512
        self.layer10 = _residual_layer('layer10', 'self.layer9', 1, 32, 32) # -> 512x512
        self.layer11 = _residual_layer('layer11', 'self.layer10', 1, 32, 32) # -> 512x512
        self.layer12 = _residual_layer('layer12', 'self.layer11', 1, 32, 32) # -> 512x512
        self.layer13 = _residual_layer('layer13', 'self.layer12', 1, 32, 32) # -> 512x512
        self.layer14 = _residual_layer('layer14', 'self.layer13', 1, 32, 32) # -> 512x512

        self.layer15 = _conv_layer('layer15', 'self.layer14', 2, 32, 64) # -> 256x256
        self.layer16 = _residual_layer('layer16', 'self.layer15', 1, 64, 64) # -> 256x256
        self.layer17 = _residual_layer('layer17', 'self.layer16', 1, 64, 64) # -> 256x256
        self.layer18 = _residual_layer('layer18', 'self.layer17', 1, 64, 64) # -> 256x256
        self.layer19 = _residual_layer('layer19', 'self.layer18', 1, 64, 64) # -> 256x256
        self.layer20 = _residual_layer('layer20', 'self.layer19', 1, 64, 64) # -> 256x256
        self.layer21 = _residual_layer('layer21', 'self.layer20', 1, 64, 64) # -> 256x256

        self.layer22 = _conv_layer('layer22', 'self.layer21', 2, 64, 128) # -> 128x128
        self.layer23 = _residual_layer('layer23', 'self.layer22', 1, 128, 128) # -> 128x128
        self.layer24 = _residual_layer('layer24', 'self.layer23', 1, 128, 128) # -> 128x128
        self.layer25 = _residual_layer('layer25', 'self.layer24', 1, 128, 128) # -> 128x128
        self.layer26 = _residual_layer('layer26', 'self.layer25', 1, 128, 128) # -> 128x128
        self.layer27 = _residual_layer('layer27', 'self.layer26', 1, 128, 128) # -> 128x128
        self.layer28 = _residual_layer('layer28', 'self.layer27', 1, 128, 128) # -> 128x128

        # self.decoder1_inputs = tf.concat([self.layer28, self.layer22], axis=3)
        # torch.cat对应的axis = 1，tensor的排列方式不同
        self.decoder1 = _conv_layer('decoder1', 'self.decoder1_inputs', 2, 2 * 128, 64,
                                         {'transpose': True})  # -> 256x256
        # self.decoder2_inputs = tf.concat([self.decoder1, self.layer15, self.layer21], axis=3)
        self.decoder2 = _conv_layer('decoder2', 'self.decoder2_inputs', 2, 3 * 64, 32,
                                         {'transpose': True})  # -> 512x512
        # self.decoder3_inputs = tf.concat([self.decoder2, self.layer8, self.layer14], axis=3)
        self.decoder3 = _conv_layer('decoder3', 'self.decoder3_inputs', 2, 3 * 32, 16,
                                         {'transpose': True})  # -> 1024x1024

        self.initial_outputs = _conv_layer('initial_outputs', 'self.decoder3', 1, 16, num_classes,
                                                {'activation': 'none', 'batchnorm': False})  # -> 1024x1024

        if num_classes == 1:
           self.last_activation = torch.nn.Sigmoid()
        else:
           self.last_activation = torch.nn.Softmax(dim=1)
    def forward(self, x):
        # encoder
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.layer6(out)
        out = self.layer7(out)

        layer8 = self.layer8(out)
        out = self.layer9(layer8)
        out = self.layer10(out)
        out = self.layer11(out)
        out = self.layer12(out)
        out = self.layer13(out)
        layer14 = self.layer14(out)

        layer15 = self.layer15(layer14)
        out = self.layer16(layer15)
        out = self.layer17(out)
        out = self.layer18(out)
        out = self.layer19(out)
        out = self.layer20(out)
        layer21 = self.layer21(out)

        layer22 = self.layer22(layer21)
        out = self.layer23(layer22)
        out = self.layer24(out)
        out = self.layer25(out)
        out = self.layer26(out)
        out = self.layer27(out)
        layer28 = self.layer28(out)

        # decoder
        decoder1_inputs = torch.cat([layer28, layer22], axis=1)
        decoder1 = self.decoder1(decoder1_inputs)
        decoder2_inputs = torch.cat([decoder1, layer15, layer21], axis=1)
        decoder2 = self.decoder2(decoder2_inputs)
        decoder3_inputs = torch.cat([decoder2, layer8, layer14], axis=1)
        decoder3 = self.decoder3(decoder3_inputs)

        initial_outputs = self.initial_outputs(decoder3)
        pre_outputs = self.last_activation(initial_outputs)
        return pre_outputs

if __name__ == '__main__':

    # MY RESNET
    Model_obj = Model()

    # print(self_encoder1)
    summary(Model_obj.cuda(), (3, 512, 512))
