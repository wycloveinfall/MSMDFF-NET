# -*-coding:utf-8-*-
# !/usr/bin/env python
# @Time    : 2023/3/7 下午4:13
# @Author  : 王玉川 uestc
# @File    : __init__.py.py
# @Description :
# Simgoid()作为网络输出层的激活函数（二分类), 应用binary_crossentropy 配合使用。
# Softmax()作为网络输出层的激活函数（多分类), 应用categorical_crossentropy() 配合使用
# 默认道路分割全部使用sigmod
'''
__call__
forward
代用的__call__的时候会先去调用forward
单独使用两者中的一个 效果是相同的
'''

if __name__ == '__main__':
    __name__='loss_fct.abc'

# soft 意思是平均的 意思，
from .dice_bce_loss import dice_bce_loss
#dice loss 最先在Vnet稳重中被提及，最后广泛用于医学图像分割
from .dice_loss import DiceLoss
# 均方误差 mean square error
from .MSE_loss import MSE_loss
