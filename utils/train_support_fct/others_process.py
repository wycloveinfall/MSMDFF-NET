# -*-coding:utf-8-*-
# !/usr/bin/env python
# @Time    : 2023/3/9 上午11:06
# @Author  : 王玉川 uestc
# @File    : others_process.py
# @Description :
from pathlib import Path
import sys
#import pandas as pd
#import numpy as np
def getDatapath(image_dir, dir_command = "*/",key_word :'str'= None):
    ''' 功能：返回指定路径下 指定深度下 含有指定关键词的文件
    image_dir:   输入的目标文件夹
    dir_command: 含有几个※号，就是下面几层目录开始
    key_word:限定路径中含有的字符,如'SRC','label'等

    fix：为了兼容，Massachusetts Roads data，将第一行内容改为第二行
    all_path = [str(iterm) for iterm in path_list if (iterm.suffix =='.tif')]
    all_path = [str(iterm) for iterm in path_list]
    '''
    Path_object = Path(image_dir)
    path_list = list(Path_object.rglob(dir_command))
    # all_path = [str(iterm) for iterm in path_list if (iterm.suffix =='.tif')]
    all_path = [str(iterm) for iterm in path_list]
    if key_word == None: pass
    else: all_path = [items for items in all_path if key_word in str(Path(items).name)]
    all_path.sort()
    return all_path

def data_check_v1(data_list, src_root='', label_root=''):
    '''
    :fct 用来检查 datalist 中是否存在异常不匹配的数据
    :param data_list:
    :return:PRE
    检查原因：ERROR 出现错误
    4: /media/gNs/DNEPLEARNING_DATA/数据仓库/公开的道路数捺/dexpglobe-road-extraction-datasetd/train/787694_sat.jpg: No such file or directory
    增加功能：匹配文件名，匹配文件的父目录
    date：2022.072
    '''
    error = []
    for index, _mess in enumerate(data_list):
        image_mess = _mess.split(' @')[0]
        label_mess = _mess.split(' @')[1]
        if not (str(Path(image_mess).parent.parent) == str(Path(label_mess).parent.parent) and
                Path(image_mess).name.split('_')[0] == Path(label_mess).name.split('_')[0]):
            error.append([index, image_mess, label_mess])
    if not error == []:
        print('data ERROR  !!!')
        print(error[0])
        sys.exit(1)
    else:
        print('data check sucess !!!')

import os
def mkdir(path):
    isExists = os.path.exists(path)
    # 判断结果
    if not isExists:
        os.makedirs(path)
        print(path + ' 创建成功\n')
        return True
    else:
        # print(path + ' 目录已存在')
        return False

# csv保存函数
