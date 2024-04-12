# -*-coding:utf-8-*-
# !/usr/bin/env python
# @Time    : 2023/3/7 下午8:31
# @Author  : 王玉川 uestc
# @File    : Soft_Iou_loss.py
# @Description :

import torch
import torch.nn as nn
import torch.nn.functional as F

class SoftIoULoss(nn.Module):
    def __init__(self, n_classes):
        super(SoftIoULoss, self).__init__()
        self.n_classes = n_classes

    @staticmethod
    def to_one_hot(tensor, n_classes):
        n, h, w = tensor.size()
        one_hot = torch.zeros(n, n_classes, h, w).scatter_(1, tensor.view(n, 1, h, w), 1)
        return one_hot

    def forward(self, input, target):
        # logit => N x Classes x H x W
        # target => N x H x W

        N = len(input)

        pred = F.softmax(input, dim=1)
        target_onehot = self.to_one_hot(target, self.n_classes)

        # Numerator Product
        inter = pred * target_onehot
        # Sum over all pixels N x C x H x W => N x C
        inter = inter.view(N, self.n_classes, -1).sum(2)

        # Denominator
        union = pred + target_onehot - (pred * target_onehot)
        # Sum over all pixels N x C x H x W => N x C
        union = union.view(N, self.n_classes, -1).sum(2)

        loss = inter / (union + 1e-16)

        # Return average loss over classes and batch
        return -loss.mean()


def soft_iou_loss(outputs, targets, smooth=1):
    # 求交集
    intersection = torch.sum(targets * outputs)
    # 求并集
    union = torch.sum(targets) + torch.sum(outputs) - intersection
    # 计算 Soft IoU Loss
    loss = (intersection + smooth) / (union + smooth)
    # 取相反数，因为优化器默认最小化损失函数
    return 1 - loss