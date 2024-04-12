# -*-coding:utf-8-*-
# !/usr/bin/env python
# @Time    : 2023/3/8 下午5:16
# @Author  : 王玉川 uestc
# @File    : data_process.py
# @Description :
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Created by: Wang Yuchuan
## School of Automation Engineering, University of Electronic Science and Technology of China
## Email: wang-yu-chuan@foxmail.com
## Copyright (c) 2017
## for paper: A Multiscale and Multidirection Feature Fusion Network for Road Detection From Satellite Imagery
# https://ieeexplore.ieee.org/document/10477437)
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
import os
import sys
import math
import numpy as np
import cv2
from osgeo import gdal
import torch
from pathlib import Path

def get_one_hot(label, N):
    '''功能:
       将标签转换成 one-hot变量  输入的类型： torch.float int  B * 1 * W * H
                              　输出类型：　B * ClassNum * W * H
    '''
    label = torch.LongTensor(label)  #
    label = label.squeeze()
    size = list(label.size())
    label = label.contiguous().view(-1)  # reshape 为向量
    ones = torch.sparse.torch.eye(N)
    ones = ones.index_select(0, label)  # 用上面的办法转为换one hot
    size.append(N)  # 把类别输目添到size的尾后，准备reshape回原来的尺寸
    ones = ones.view(*size).transpose(2, 1).transpose(1, 0)  # rshape成 N * C* W * H
    # cv2.imwrite('003.png',ones[2].data.numpy()*255)
    return ones
#### base image read ####################################################
#Tif文件读取
def ReadTifImg(filename):
    '''功能：用于读取TIF格式的遥感图像，
       返回值：im_proj : 地图投影信息，一般在剪裁，拼合图像的时候不修改这部分信息
             im_geotrans : 仿射矩阵,里面存放了地图绝对的地理信息位置
             im_data：通道顺序位 [channel，width,height]'''
    dataset = gdal.Open(filename)  # 打开文件

    im_width = dataset.RasterXSize  # 栅格矩阵的列数
    im_height = dataset.RasterYSize  # 栅格矩阵的行数

    im_geotrans = dataset.GetGeoTransform()  # 仿射矩阵
    im_proj = dataset.GetProjection()  # 地图投影信息

    # GeoTransform[0]  横向 水平 [0,0.5,0,0,0,-0.5]
    # GeoTransform[3]  左上角位置
    # GeoTransform[1]是像元宽度 正值 相加
    # GeoTransform[5]是像元高度 负值 相减
    # 如果影像是指北的,GeoTransform[2]和GeoTransform[4]这两个参数的值为0。
    im_data = dataset.ReadAsArray(0, 0, im_width, im_height)  # 将数据写成数组，对应栅格矩阵
    del dataset
    return im_proj, im_geotrans, im_data

#Tif文件写入
def WriteTifImg(filename, im_proj, im_geotrans, im_data, datatype=None):
    '''功能：用于写TIF格式的遥感图像，同时兼容一个通道 和 三个通道
       返回值：im_proj : 地图投影信息，保持与输入图像相同
             im_geotrans : 仿射矩阵,计算当前图像块的仿射信息
             im_data：通道顺序位 [channel,height,width]， 当前图像块的像素矩阵，
             datatype：指定当前图像数据的数据类型，默认和输入的im_data类型相同'''

    # gdal数据类型包括
    # gdal.GDT_Byte,
    # gdal .GDT_UInt16, gdal.GDT_Int16, gdal.GDT_UInt32, gdal.GDT_Int32,
    # gdal.GDT_Float32, gdal.GDT_Float64

    # 判断栅格数据的数据类型
    if datatype is None:  # im_data.dtype.name数据格式
        if 'int8' in im_data.dtype.name:
            datatype = gdal.GDT_Byte
        elif 'int16' in im_data.dtype.name:
            datatype = gdal.GDT_UInt16
        else:
            datatype = gdal.GDT_Float32

    # 判读数组维数
    if len(im_data.shape) == 3:
        im_bands, im_height, im_width = im_data.shape
    else:
        im_bands, (im_height, im_width) = 1, im_data.shape

        # 创建文件
    driver = gdal.GetDriverByName("GTiff")  # 数据类型必须有，因为要计算需要多大内存空间
    dataset = driver.Create(filename, im_width, im_height, im_bands, datatype)

    dataset.SetGeoTransform(im_geotrans)  # 写入仿射变换参数
    dataset.SetProjection(im_proj)  # 写入投影

    if im_bands == 1:
        dataset.GetRasterBand(1).WriteArray(im_data)  # 写入数组数据
    else:
        for i in range(im_bands):  # 按波段写入
            dataset.GetRasterBand(i + 1).WriteArray(im_data[i])

    del dataset






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

def key_sort_for_SpaceNet(data_list):

    result = sorted(data_list, key=lambda x: (x.split('/')[-1].split('_')[-4],
                                                x.split('/')[-1].split('img')[-1].split('_')[0].split('.')[0]))
    return result

def data_check_for_Massachusetts(data_list, src_root='', label_root=''):
    '''
    :fct 用来检查 datalist 中是否存在异常不匹配的数据
    :param data_list:
    :return:PRE
    检查原因：ERROR 出现错误
    增加功能：匹配文件名，匹配文件的父目录
    date：2022.072
    '''
    error = []
    for index, _mess in enumerate(data_list):
        image_mess = _mess.split(' @')[0]
        label_mess = _mess.split(' @')[1]
        if not (str(Path(image_mess).parent.parent) == str(Path(label_mess).parent.parent) and
                image_mess.split('/')[-1].split('.')[0].replace('_sat','') ==
                label_mess.split('/')[-1].split('.')[0].replace('_mask','')):
            error.append([index, image_mess, label_mess])
    if not error == []:
        print('data ERROR  !!!')
        print(error[0])
        sys.exit(1)
    else:
        print('data check sucess !!!')

def data_check_for_SpaceNet(data_list, src_root='', label_root=''):
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
                image_mess.split('/')[-1].split('img')[-1].split('_')[0].split('.')[0] ==
                label_mess.split('/')[-1].split('img')[-1].split('_')[0].split('.')[0]):
            error.append([index, image_mess, label_mess])
    if not error == []:
        print('data ERROR  !!!')
        print(error[0])
        sys.exit(1)
    else:
        print('data check sucess !!!')

def data_check_v2(image_data_list, label_data_list):
    '''
    fct 用来检查 datalist 中是否存在异常不匹配的数据
    :param image_data_list:
    :param label_data_list:
    :return:
    '''
    error = []
    for index, (image_mess, label_mess) in enumerate(zip(image_data_list, label_data_list)):
        if not image_mess.split(' @')[1:] == label_mess.split(' @')[1:]:
            error.append([index, image_mess, label_mess])
    if not error == []:
        print('data ERROR  !!!')
        print(error[0])
        sys.exit(1)
    else:
        print('data check sucess !!!')

def txt_to_list(txt_Dir):
    train_list = []
    f = open(txt_Dir,'r+',encoding='utf-8')
    for txt in f:
        train_list.append(txt[:-1])
    f.close()
    return train_list

def message_to_img_v1(message:'str'):
    '''
    校验者：wyc 2022.02.23
    将生成记录转换为矩阵图片、地理信息并生成ID
    '''
    src_path = message.split(' @')[0]
    lab_path = message.split(' @')[1]
    src_proj, src_geotrans, src_img = ReadTifImg(src_path)
    lab_proj, lab_geotrans, lab_img = ReadTifImg(lab_path)
    imageName = src_path.split('/')[-1].split('.')[0]

    return src_img,lab_img,imageName
