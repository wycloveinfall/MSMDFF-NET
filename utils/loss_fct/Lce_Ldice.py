# -*-coding:utf-8-*-
import numpy as np
import torch
import torch.nn as nn
from torch.nn.modules.loss import CrossEntropyLoss
# from .dice_bce_loss import dice_bce_loss

class DiceLoss(nn.Module):
    def __init__(self, n_classes):
        super(DiceLoss, self).__init__()
        self.n_classes = n_classes

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i  # * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob.unsqueeze(1))
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _dice_loss(self, score, target):
        target = target.float()
        smooth = 1e-5
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        return loss

    def forward(self, inputs, target, weight=None, softmax=False):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)#12, 6, 256, 256
        # target = self._one_hot_encoder(target)#[12, 6, 256, 256]
        if weight is None:
            weight = [1] * self.n_classes
        assert inputs.size() == target.size(), 'predict {} & target {} shape do not match'.format(inputs.size(), target.size())
        class_wise_dice = []
        loss = 0.0
        for i in range(0, self.n_classes):
            dice = self._dice_loss(inputs[:, i], target[:, i])
            class_wise_dice.append(1.0 - dice.item())
            loss += dice * weight[i]
        return loss / self.n_classes


class Lce_Ldice_loss(object):
    def __init__(self,num_classes=1):
        # self.ce_loss = CrossEntropyLoss(reduction='none')
        self.ce_loss = CrossEntropyLoss()
        self.dice_loss = DiceLoss(num_classes)

    # def __call__(self, outputs, label_batch):
    #     # loss_ce = self.ce_loss(outputs, label_batch[:].long())
    #     _loss_ce = self.ce_loss(outputs, label_batch)
    #     loss_ce = _loss_ce.mean([1, 2]).view(label_batch.size(0), label_batch.size(1))
    #     loss_dice = self.dice_loss(outputs, label_batch, softmax=True)
    #     loss = 0.5 * loss_ce + 0.5 * loss_dice  # loss
    #     # _c_tensor = torch.from_numpy(np.ones([6,1])).cuda()
    #     return loss_ce, loss
    def __call__(self, outputs, label_batch):
        # loss_ce = self.ce_loss(outputs, label_batch[:].long())
        loss_ce = self.ce_loss(outputs, label_batch)
        loss_dice = self.dice_loss(outputs, label_batch, softmax=True)
        loss = 0.5 * loss_ce + 0.5 * loss_dice  # loss
        _c_tensor = torch.tensor([loss,loss,loss,loss,loss,loss]).cuda()
        return _c_tensor, loss
