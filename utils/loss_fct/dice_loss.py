# -*-coding:utf-8-*-
# !/usr/bin/env python
# @Time    : 2023/3/7 下午7:58
# @Author  : 王玉川 uestc
# @File    : dice_loss.py
# @Description :
'''
dice_loss_src 是源码 2023.03.08已经比对原公式正确
dice_loss 是修改后的，目的是按batch 输出loss
dice loss 最先在Vnet稳重中被提及，最后广泛用于医学图像分割
'''
import torch
import torch.nn as nn

class DiceLoss_src(nn.Module):  # 测试 loss 函数
    '''support: https://zhuanlan.zhihu.com/p/86704421
    '''

    def __init__(self):
        super(DiceLoss_src, self).__init__()

    def forward(self, input, target):
        '''
        inpu_size: batch * w * h
        '''
        N = target.size(0)
        smooth = 1

        input_flat = input.view(N, -1)  # 按照规则 重新排列
        target_flat = target.view(N, -1)

        intersection = input_flat * target_flat

        loss = 2 * (intersection.sum(1) + smooth) / (input_flat.sum(1) + target_flat.sum(1) + smooth)
        loss = 1 - loss.sum() / N

        return loss

class DiceLoss(nn.Module):  # 训练的loss
    def __init__(self, bacth_metric = True):
        super(DiceLoss, self).__init__()
        self.bacth_metric = bacth_metric
    def _dice_coeff(self, target, predict):
        '''
        #这个也算Iou的一种度量方式，当i，j完全重合的时候为1
        '''
        smooth = 10e-8 # may change
        if not self.bacth_metric:
            i = torch.sum(target)
            j = torch.sum(predict)
            intersection = torch.sum(target * predict)
        else:
            i = target.sum(1).sum(1).sum(1)
            j = predict.sum(1).sum(1).sum(1)
            intersection = (target * predict).sum(1).sum(1).sum(1)
        score = (2. * intersection + smooth) / (i + j + smooth) #去掉+0.1
        # Iou = (intersection + 0.1 * smooth) / (i + j - intersection + smooth)
        return score.view(target.size(0), target.size(1))

    def _dice_loss(self, y_true, y_pred):
        # loss = 10 - 2 * self.soft_dice_coeff(y_true, y_pred)
        loss = 1 - self._dice_coeff(y_true, y_pred)
        return loss

    def __call__(self, y_pred, y_true):
        bb = self._dice_loss(y_true, y_pred)
        b = torch.mean(bb)

        return bb, b



