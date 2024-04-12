# -*-coding:utf-8-*-
# !/usr/bin/env python
# @Time    : 2023/3/7 下午8:12
# @Author  : 王玉川 uestc
# @File    : MSE_loss.py
# @Description :
'''
均方误差 mean square error
'''
import torch
import torch.nn as nn

class MSE_loss(nn.Module):  # 训练的loss
    def __init__(self):
        super(MSE_loss, self).__init__()
        self.mse_loss = nn.MSELoss(reduction='none')
    def __call__(self, y_pred, y_true):
        loss_get = self.mse_loss(y_pred, y_true)
        return loss_get.mean([1, 2, 3]), loss_get.mean([0, 1, 2, 3])


if __name__ == "__main__":
    import numpy as np
    np.random.seed(10)

    # 构造loss变量
    class_num = 1
    _pred = np.random.random((2, class_num, 512, 512)) #假设这个是 卷积网络最后一侧输出 分类数为1 用sigmod
    pred_tensor = torch.tensor(_pred).to(torch.float32)                  #分类数大于1用softmax
    pred_ = torch.sigmoid(pred_tensor) if class_num == 1 else torch.softmax(pred_tensor,dim=1)
    '''
    pred_[0][0][1][1]
    Out[8]: tensor(0.6110, dtype=torch.float64)
    pred_[0][1][1][1]
    Out[9]: tensor(0.3890, dtype=torch.float64)
    '''
    if class_num == 1:
        target = np.random.randint(0, 2, size=(2,class_num,512, 512))
    else:
        target1 = np.random.randint(0, 2, size=(2,512, 512))
        target1 = target1[np.newaxis, :, :]
        target2 = np.random.randint(0, 2, size=(2,512, 512))
        target2 = target2[np.newaxis, :, :]
        target = np.vstack([target1, target2])

    target = torch.tensor(target).to(torch.float32)

    F1 = MSE_loss() #带batch的loss
    F2 = nn.MSELoss()
    loss1 = F1(pred_,target)
    loss2 = F2(pred_,target)
    print(loss1)
    print(loss2)
