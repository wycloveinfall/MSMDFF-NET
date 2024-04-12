import torch
import torch.nn as nn

import cv2
import numpy as np
import torch.nn.functional as F

"""
https://blog.csdn.net/qq_36201400/article/details/109367086
https://zhuanlan.zhihu.com/p/101773544
"""
__all__ = ['dice_bce_loss']



class dice_bce_loss_edges(nn.Module):  # 训练的loss
    def __init__(self,bacth_metric=True):
        super(dice_bce_loss_edges, self).__init__()

        self.bce_loss = nn.BCELoss(reduction='none')
        self.lamda2 = torch.tensor(1)

        self.bacth_metric = bacth_metric

    def tensor_erode(self,bin_img, ksize=3): # 已测试 腐蚀可用
        # 首先为原图加入 padding，防止腐蚀后图像尺寸缩小
        B, C, H, W = bin_img.shape
        pad = (ksize - 1) // 2
        bin_img = F.pad(bin_img, [pad, pad, pad, pad], mode='constant', value=0)
        # 将原图 unfold 成 patch
        patches = bin_img.unfold(dimension=2, size=ksize, step=1)
        patches = patches.unfold(dimension=3, size=ksize, step=1)
        # B x C x H x W x k x k
        # 取每个 patch 中最小的值，i.e., 0
        eroded, _ = patches.reshape(B, C, H, W, -1).min(dim=-1)
        return eroded

    def get_iou(self,outputs: torch.Tensor, labels: torch.Tensor, SMOOTH=1e-7):  # todo
        '''
        You can comment out this line if you are passing tensors of equal shape
        But if you are passing output from UNet or something it will most probably
        be with the BATCH x 1 x H x W shape
        outputs = y_pred
        labels = y_true
        '''
        with torch.no_grad():
            outputs = outputs.cpu().data.numpy().squeeze()  # BATCH x 1 x H x W => BATCH x H x W
            labels = labels.cpu().data.numpy().squeeze()
            # test    labels = labels.squeeze(1)
            if len(outputs.shape) == 3:
                intersection = (outputs & labels).float().sum((1, 2))  # Will be zero if Truth=0 or Prediction=0
                union = (outputs | labels).float().sum((1, 2))  # Will be zzero if both are 0
            else:
                intersection = (outputs & labels).float().sum((2, 3))  # Will be zero if Truth=0 or Prediction=0
                union = (outputs | labels).float().sum((2, 3))  # Will be zzero if both are 0

            iou = (intersection + SMOOTH * 0.4) / (union + SMOOTH)  # We smooth our devision to avoid 0/0
            # thresholded = torch.clamp(20 * (iou - 0.5), 0, 10).ceil() / 10  # This is equal to comparing with thresolds

        return iou  # Or thresholded.mean() if you are interested in average across the batch

    def get_circle_weights(self,y_pred, y_true):
        with torch.no_grad():
            erode1 = tensor_erode(y_true, ksize=3)
            erode2 = tensor_erode(erode1, ksize=3)
            erode3 = tensor_erode(erode2, ksize=3)
            erode4 = tensor_erode(erode3, ksize=3)
            circle_1 = y_true - erode1
            circle_2 = erode1 - erode2
            circle_3 = erode2 - erode3
            circle_4 = erode3 - erode4

            circle_1 = circle_1*0.5
            circle_2 = circle_2*0.6
            circle_3 = circle_3*0.7
            circle_4 = circle_4*0.8

            circle_all = circle_1 + circle_2 +circle_3 +circle_4
            circle_all[circle_all==0] = 1

            # for y_t in y_true:
            #     y_weights.append(0 if y_t.sum([0,1,2])==0 else 1/y_t.sum([0,1,2]))


        return circle_all
        '''
        import imageio
        import numpy as np
        t_data = y_pred.cpu().data.numpy()*255

        t_data = t_data.astype(np.uint8)
        imageio.imwrite('/home/gis/下载/测试边缘/src_im.tif',t_data[0][0])
        
        for index,circle_ in enumerate([circle_1,circle_2,circle_3]):
            circle_11 = circle_.cpu().data.numpy()*255
            imageio.imwrite('/home/gis/下载/测试边缘/01/c_%d.tif'%(index+1),circle_11.astype(np.uint8)[0][0])
        
        circle_all1 = circle_all.cpu().data.numpy()*255
        imageio.imwrite('/home/gis/下载/测试边缘/c_all.tif',circle_all1.astype(np.uint8)[0][0])
        '''
    def get_circle_weights0907(self,y_pred,y_true):
        with torch.no_grad():
            erode1 = tensor_erode(y_pred, ksize=3)
            erode2 = tensor_erode(erode1, ksize=3)
            circle_1 = y_pred - erode1
            circle_2 = erode1 - erode2

            circle_1[circle_2>0.20] = 0 #去掉内环
            circle_1[circle_1<0.1] = 0  #去掉非边界

            circle_2[y_pred<=0.5] = 1
            circle_2[y_pred>=0.5] = 1
            circle_2[circle_1 >= 0.1] = 0  # 清理一个环

            circle_3 = circle_1+circle_2
            # circle_3[circle_3>=0.99] = 1

        return circle_3

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
        self.score = (2. * intersection + smooth) / (i + j + smooth) #去掉+0.1
        # Iou = (intersection + 0.1 * smooth) / (i + j - intersection + smooth)
        return self.score.view(target.size(0), target.size(1))

    def get_loss_weghts(self):
        with torch.no_grad():
            loss_weghts = self.score.detach()
        return loss_weghts

    def soft_dice_coeff_old(self, target, predict):
        '''
        说明：该版本已废弃，由于创建了新的变量，导致数据被拷贝到cpu，使训练过程产生瓶颈
        '''
        smooth = 10e-8  # may change
        iou = torch.zeros(target.size(0), target.size(1))

        for index in range(target.size(1)):
            intersection = (target[:, index, :, :] * predict[:, index, :, :]).float().sum((1, 2))
            union = (target[:, index, :, :] + predict[:, index, :, :]).float().sum((1, 2))
            # iou[:, index] = (intersection + smooth) / (union - intersection + smooth)
            iou[:, index] = (2.*intersection + smooth) / (union + smooth)
        score = iou  # 点乘
        return score

    def soft_dice_loss(self, y_true, y_pred):
        # loss = 10 - 2 * self.soft_dice_coeff(y_true, y_pred)
        loss = 1 - self.soft_dice_coeff(y_true, y_pred)
        return loss

    def __call__(self, y_pred, y_true):
        # 测试输入：torch.Size([2, 4, 512, 512])
        # circle_weghts = self.get_circle_weights(y_pred, y_true) #0907以前
        circle_weghts = self.get_circle_weights0907(y_pred,y_true) #0907
        loss_bce = torch.mul(self.bce_loss(y_pred, y_true),circle_weghts)

        aa = loss_bce.mean([1,2,3]).view(y_true.size(0), y_true.size(1))
        bb = self.soft_dice_loss(y_true, y_pred)

        a = torch.mean(aa)
        b = torch.mean(bb) #0.5078

        return self.lamda2 * aa+bb, self.lamda2 * a + b


class MulticlassDiceLoss(nn.Module):
    """
    requires one hot encoded target. Applies DiceLoss on each class iteratively.
    requires input.shape[0:1] and target.shape[0:1] to be (N, C) where N is
      batch size and C is number of classes
    """
    def __init__(self):
        super(MulticlassDiceLoss, self).__init__()
 
    def forward(self, input, target, weights=None):
 
        C = target.shape[1]
 
        # if weights is None:
        #   weights = torch.ones(C) #uniform weights for all classes
 
        dice = DiceLoss()
        totalLoss = 0
 
        for i in range(C):
            diceLoss = dice(input[:,i], target[:,i])
            if weights is not None:
                diceLoss *= weights[i]
            totalLoss += diceLoss

        return totalLoss
    



class MSE_loss_edges(nn.Module):  # 训练的loss
    def __init__(self):
        super(MSE_loss_edges, self).__init__()
        self.mse_loss = nn.MSELoss(reduction='none')

    def tensor_erode(self,bin_img, ksize=3): # 已测试 腐蚀可用
        # 首先为原图加入 padding，防止腐蚀后图像尺寸缩小
        B, C, H, W = bin_img.shape
        pad = (ksize - 1) // 2 # “//”是一个算术运算符，表示整数除法，它可以返回商的整数部分（向下取整）
        bin_img = F.pad(bin_img, [pad, pad, pad, pad], mode='constant', value=0)
        # 将原图 unfold 成 patch
        patches = bin_img.unfold(dimension=2, size=ksize, step=1)
        patches = patches.unfold(dimension=3, size=ksize, step=1)
        # B x C x H x W x k x k
        # 取每个 patch 中最小的值，i.e., 0
        eroded, _ = patches.reshape(B, C, H, W, -1).min(dim=-1)
        return eroded

    def tensor_dilate(self,bin_img, ksize=3): #
        # 首先为原图加入 padding，防止图像尺寸缩小
        B, C, H, W = bin_img.shape
        pad = (ksize - 1) // 2
        bin_img = F.pad(bin_img, [pad, pad, pad, pad], mode='constant', value=0)
        # 将原图 unfold 成 patch
        patches = bin_img.unfold(dimension=2, size=ksize, step=1)
        patches = patches.unfold(dimension=3, size=ksize, step=1)
        # B x C x H x W x k x k
        # 取每个 patch 中最小的值，i.e., 0
        dilate, _ = patches.reshape(B, C, H, W, -1).max(dim=-1)
        return dilate

    def get_circle_weights(self,y_pred, y_true):
        with torch.no_grad():
            erode1 = tensor_erode(y_true, ksize=3)
            erode2 = tensor_erode(erode1, ksize=3)
            erode3 = tensor_erode(erode2, ksize=3)
            erode4 = tensor_erode(erode3, ksize=3)
            circle_1 = y_true - erode1
            circle_2 = erode1 - erode2
            circle_3 = erode2 - erode3
            circle_4 = erode3 - erode4

            circle_1 = circle_1*0.4
            circle_2 = circle_2*0.5
            circle_3 = circle_3*0.6
            circle_4 = circle_4*0.7

            circle_all = circle_1 + circle_2 + circle_3 + circle_4
            circle_all[circle_all==0] = 0.8

        return circle_all


    def __call__(self, y_pred, y_true):
        circle_weghts = self.get_circle_weights(y_pred, y_true)
        torch.cuda.empty_cache() #释放显存

        loss_get = self.mse_loss(y_pred, y_true)
        loss_get_ = torch.mul(loss_get,circle_weghts)



        # return loss_get.mean([1, 2, 3]), loss_get.mean([0, 1, 2, 3])
        return loss_get_.mean([1, 2, 3]), loss_get_.mean([0, 1, 2, 3])



# 测试
def tensor_erode(bin_img, ksize=3):
    # 首先为原图加入 padding，防止腐蚀后图像尺寸缩小
    B, C, H, W = bin_img.shape
    pad = (ksize - 1) // 2
    bin_img = F.pad(bin_img, [pad, pad, pad, pad], mode='constant', value=0)
    # 将原图 unfold 成 patch
    patches = bin_img.unfold(dimension=2, size=ksize, step=1)
    patches = patches.unfold(dimension=3, size=ksize, step=1)
    # B x C x H x W x k x k
    # 取每个 patch 中最小的值，i.e., 0
    eroded, _ = patches.reshape(B, C, H, W, -1).min(dim=-1)
    return eroded

