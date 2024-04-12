# -*-coding:utf-8-*-
# !/usr/bin/env python
# @Time    : 2023/3/9 上午10:02
# @Author  : 王玉川 uestc
# @File    : model_name_process.py
# @Description :
import os
import numpy as np
def loadModelNames(modelDir='./weights/'):
    '''
    用来获取，文件夹下 最新的 模型文件的 名字
    '''
    model_list = os.listdir(modelDir)
    model_num = [float(index.split('_')[0]) for index in model_list]
    model_num = np.array(model_num)
    model_num = np.sort(model_num)
    try:
        max_num = model_num.max()
        # model_name = []
        for dir_context in model_list:
            if str(max_num) in dir_context:
                model_name = modelDir + dir_context
                # print('load model is[$]: ', dir_context)
                break
        epoch = int(model_name.split('$')[1])
        lr_current = float(model_name.split('$')[-2])
    except:
        max_num = 1
        model_name = 'error'
        epoch = 0
        lr_current = None

    return model_name, max_num, epoch, lr_current

def loadModelNames_by_keys(modelDir='./weights/',keys= None):
    '''
    用来获取，文件夹下 最新的 模型文件的 名字
    '''
    model_list = os.listdir(modelDir)
    model_num = [float(index.split('_')[0]) for index in model_list]
    model_num = np.array(model_num)
    model_num = np.sort(model_num)
    try:
        if keys == None:
           max_num = model_num.max()
        else:
           max_num = keys
        # model_name = []
        for dir_context in model_list:
            if str(max_num) == dir_context.split('$')[1]:
                model_name = modelDir + dir_context
                # print('load model is[$]: ', dir_context)
                break
        epoch = int(model_name.split('$')[1])
        lr_current = float(model_name.split('$')[-2])
    except:
        max_num = 1
        model_name = 'error'
        epoch = 0
        lr_current = None

    return model_name, max_num, epoch, lr_current


