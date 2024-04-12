# -*-coding:utf-8-*-
# !/usr/bin/env python
# @Time    : 2023/3/7 下午4:22
# @Author  : 王玉川 uestc
# @File    : dice_bce_loss.py
# @Description :
'''
dice_bce_loss_src 是源码
dice_bce_loss 是修改后的，目的是按batch 输出loss
两个戴拿已经过测试 输出结果可靠

'''

import torch
import torch.nn as nn

class dice_bce_loss_src(nn.Module):
    def __init__(self, batch=True):
        super(dice_bce_loss_src, self).__init__()
        self.batch = batch
        self.bce_loss = nn.BCELoss()

    def _dice_coeff(self, y_true, y_pred):
        smooth = 10e-8  # may change
        if self.batch:
            i = torch.sum(y_true)
            j = torch.sum(y_pred)
            intersection = torch.sum(y_true * y_pred)
        else:
            i = y_true.sum(1).sum(1).sum(1)
            j = y_pred.sum(1).sum(1).sum(1)
            intersection = (y_true * y_pred).sum(1).sum(1).sum(1)
        score = (2. * intersection + smooth) / (i + j + smooth)
        # score = (intersection + smooth) / (i + j - intersection + smooth)#iou
        return score.mean()

    def _dice_loss(self, y_true, y_pred):
        loss = 1 - self._dice_coeff(y_true, y_pred)
        return loss

    def forward(self, y_pred,y_true):
        a = self.bce_loss(y_pred, y_true)      #0.7338
        b = self._dice_loss(y_true, y_pred)#0.4458
        return a + b


class dice_bce_loss(nn.Module):  # 训练的loss
    def __init__(self, bacth_metric = True):
        super(dice_bce_loss, self).__init__()

        self.bce_loss = nn.BCELoss(reduction='none')
        self.lamda2 = torch.tensor(1)
        self.bacth_metric = bacth_metric

    def soft_dice_coeff(self, target, predict):
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

    def soft_dice_loss(self, y_true, y_pred):
        # loss = 10 - 2 * self.soft_dice_coeff(y_true, y_pred)
        loss = 1 - self.soft_dice_coeff(y_true, y_pred)
        return loss

    def __call__(self, y_pred, y_true):
        aa = self.bce_loss(y_pred, y_true)
        aa = aa.mean([1, 2, 3]).view(y_true.size(0), y_true.size(1))
        bb = self.soft_dice_loss(y_true, y_pred)
        a = torch.mean(aa)
        b = torch.mean(bb)

        return aa+ self.lamda2 * bb, a + self.lamda2 * b


