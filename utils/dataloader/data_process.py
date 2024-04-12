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
import sys
import math
import numpy as np
import cv2
from osgeo import gdal
import torch
from pathlib import Path
import os
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

def SetSize(GeoTransform: 'tuple', resize_proportion: 'float'):
    geo_w = GeoTransform[1] * resize_proportion
    geo_h = GeoTransform[5] * resize_proportion
    roate_x = GeoTransform[2]
    roate_y = GeoTransform[4]
    geo_x = GeoTransform[0]
    geo_y = GeoTransform[3]
    GeoTransform = (geo_x, geo_w, roate_x, geo_y, roate_y, geo_h)
    return GeoTransform

def SetCutting(GeoTransform:'tuple',x:'int',y:'int'):
    '''
    设置裁切后的仿射系数
    x，y是裁切区域左上角位置
    '''
    GeoTransform0 = GeoTransform[0] + x * GeoTransform[1]
    GeoTransform3 = GeoTransform[3] + y * GeoTransform[5]
    geotrans = (GeoTransform0,GeoTransform[1],GeoTransform[2],GeoTransform3,GeoTransform[4],GeoTransform[5])
    return geotrans
# 旋转后的仿射矩阵变换函数
def SetRotate(GeoTransform: 'tuple', angle: 'int' = 0, h: 'int' = 0, w: 'int' = 0):
    '''
    设置旋转后的Tif图像仿射系数，使旋转后坐标保持不变
    h，w为旋转图像矩阵的宽高
    '''
    # 仿射矩阵旋转变换
    GeoTransform1 = GeoTransform[1] * math.cos(angle / 180 * math.pi) + GeoTransform[2] * math.sin(
        angle / 180 * math.pi)
    GeoTransform2 = - GeoTransform[1] * math.sin(angle / 180 * math.pi) + GeoTransform[2] * math.cos(
        angle / 180 * math.pi)
    GeoTransform4 = GeoTransform[4] * math.cos(angle / 180 * math.pi) + GeoTransform[5] * math.sin(
        angle / 180 * math.pi)
    GeoTransform5 = - GeoTransform[4] * math.sin(angle / 180 * math.pi) + GeoTransform[5] * math.cos(
        angle / 180 * math.pi)
    # 仿射矩阵左上角起始点修正
    if angle > 0:
        angle = angle % 360 - 360
    else:
        pass
    sloping_side = math.sqrt(2 * (1 - math.cos((abs(angle)) / 180 * math.pi))) * math.sqrt(
        math.pow(h / 2, 2) + math.pow(w / 2, 2))
    beta = (180 - abs(angle)) / 2 - math.atan(w / h) / math.pi * 180
    x_ = sloping_side * math.sin(beta / 180 * math.pi)
    y_ = sloping_side * math.cos(beta / 180 * math.pi)
    GeoTransform0 = GeoTransform[0] - x_ * GeoTransform[1]
    GeoTransform3 = GeoTransform[3] + y_ * GeoTransform[5]
    GeoTransform = (GeoTransform0, GeoTransform1, GeoTransform2, GeoTransform3, GeoTransform4, GeoTransform5)
    return GeoTransform

# 按指定中心位置，指定角度，指定放缩比例，指定宽高裁切矩形区域
def area_cutting(src_proj, src_geotrans, src_img, src_h: 'int' = 0, src_w: 'int' = 0, x: 'int' = 0, y: 'int' = 0,
                 angle: 'int' = 0, size: 'float' = 1.0):
    """
    校验者：wyc 2022.02.23
    area_cutting 将输入tif图像矩阵按照输入高宽，中心点坐标，角度裁切
    # 输入tif图片信息三要素
    # angle: 逆时针旋转的角度
    # 输出tif图片仿射矩阵并进行修正

    date:2022.07.26发现对于二值图的操作有点问题 已修复
    """
    if len(src_img.shape) == 3:
        C, H, W = src_img.shape
    else:
        H, W = src_img.shape
    h = int(src_h * size)
    w = int(src_w * size)
    l: int = int(math.sqrt(math.pow(h / 2, 2) + math.pow(w / 2, 2))) + 1  # 计算切割的外接圆半径
    proj = src_proj
    geotrans = src_geotrans
    if len(src_img.shape) == 3:
        zeros_img = np.zeros((3, src_h, src_w))
    else:
        zeros_img = np.zeros((src_h, src_w))

    if angle == 0:
        x1 = x - int(h / 2)
        x2 = x + int(h / 2)
        y1 = y - int(w / 2)
        y2 = y + int(w / 2)
        img = src_img[:, y1:y2, x1:x2] if len(src_img.shape) == 3 else src_img[y1:y2, x1:x2]
        geotrans = SetCutting(geotrans, x1, y1)  # 修正因第一次裁切导致的仿射系数误差
    elif h < 0 or h > H:
        print('please input h in range(0-%d)' % H)
        img = zeros_img
    elif w < 0 or w > W:
        print('please input w in range(0-%d)' % W)
        img = zeros_img
    elif x < l or x > (W - l) :
        print('please input x in range(%d-%d) or input geo_x in range(%d-%d)' % (
            l, W - l, l * geotrans[1] + geotrans[0], (W - l) * geotrans[1] + geotrans[0]))
        img = zeros_img
    elif y < l or y > (H - l):
        print('please input y in range(%d-%d) or input geo_y in range(%d-%d)' % (
            l, H - l, (H - l) * geotrans[5] + geotrans[3], l * geotrans[5] + geotrans[3]))
        img = zeros_img
    else:
        x1 = x - l
        x2 = x + l
        y1 = y - l
        y2 = y + l
        x1_ = l - int(w / 2)
        x2_ = l + w - int(w / 2)
        y1_ = l - int(h / 2)
        y2_ = l + h - int(h / 2)
        M = cv2.getRotationMatrix2D((l, l), -angle, 1)
        # 裁切矩形的外接圆的外接正方形
        if len(src_img.shape) == 3:
            img = src_img[:, y1:y2, x1:x2]
            img = img.transpose(1, 2, 0)
            res = cv2.warpAffine(img, M, (2 * l, 2 * l), borderValue=(0, 0, 0))
            img = res[y1_:y2_, x1_:x2_, :]
            img = cv2.resize(img, (src_w, src_h), interpolation=cv2.INTER_LINEAR)
            img = img.transpose(2, 0, 1)
            geotrans = SetCutting(geotrans, x1, y1) # 修正因第一次裁切导致的仿射系数误差
            geotrans = SetCutting(geotrans, x1_, y1_) #修正因第二次裁切导致的仿射系数误差
            geotrans = SetSize(geotrans, size) # 修正因放缩导致的仿射系数误差
            geotrans = SetRotate(geotrans, -angle, src_h, src_w) # 修正因旋转导致的仿射系数误差
        else:
            img = src_img[y1:y2, x1:x2]
            res = cv2.warpAffine(img, M, (2*l, 2*l), borderValue=(0, 0, 0))
            img = res[y1_:y2_, x1_:x2_]
            img = cv2.resize(img, (src_w, src_h), interpolation=cv2.INTER_LINEAR)
            geotrans = SetCutting(geotrans, x1, y1)
            geotrans = SetCutting(geotrans, x1_, y1_)
            geotrans = SetSize(geotrans, size)
            geotrans = SetRotate(geotrans, -angle, src_h, src_w)
    return proj, geotrans, img

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
    4: /media/gNs/DNEPLEARNING_DATA/数据仓库/公开的道路数捺/dexpglobe-road-extraction-datasetd/train/787694_sat.jpg: No such file or directory
    增加功能：匹配文件名，匹配文件的父目录
    date：2022.072
    '''
    error = []
    for index, _mess in enumerate(data_list):
        image_mess = _mess.split(' @')[0]
        label_mess = _mess.split(' @')[1]
        if not (str(Path(image_mess).parent.parent) == str(Path(label_mess).parent.parent) and
                image_mess.split('/')[-1].split('.')[0] ==
                label_mess.split('/')[-1].split('.')[0]):
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


##################################################################################################################
############################# message to image ###################################################################
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

    src_x = src_geotrans[0]
    src_y = src_geotrans[3]
    lab_x = lab_geotrans[0]
    lab_y = lab_geotrans[3]
    if src_x != lab_x:
        print('x mismatch')
        id = None
    elif src_y != lab_y:
        print('y mismatch')
        id = None
    elif src_proj != lab_proj:
        print('proj mismatch')
        id = None
    elif src_geotrans != lab_geotrans:
        print('geotrans mismatch')
        id = None
    else:#以下的条件语句，是为了兼容 deeplglobal数据集
        if 'SRC' in src_path.split('\\')[-1].split('/')[-1]:
           id = src_path.split('\\')[-1].split('/')[-1].split('SRC')[0] + '_%d_%d'%(src_x,src_y)
        else:
            id = src_path.split('\\')[-1].split('/')[-1].split('_')[0] + '_%d_%d' % (src_x, src_y)
    return src_proj,src_geotrans,src_img,lab_img,imageName


def message_to_img_v2(message='', data={}, img_Dir='', mode='r'):
    '''
    校验者：wyc 2022.02.23
    功能：
    :param message:
    :param data:no use
    :param img_Dir:
    :param mode:
    :return:
    '''
    # message = src_path + ' @' + 'y,x' + ' @' + 'h,w' + ' @' + 'y_shift,x_shift,angle,size' + ' @' + 'j,i'
    # time1 = datetime.datetime.now()
    path = message.split(' @')[0]
    y = int(message.split(' @')[1].split(',')[0])
    x = int(message.split(' @')[1].split(',')[1])
    h = int(message.split(' @')[2].split(',')[0])
    w = int(message.split(' @')[2].split(',')[1])
    y_shift = int(message.split(' @')[3].split(',')[0])
    x_shift = int(message.split(' @')[3].split(',')[1])
    angle = int(message.split(' @')[3].split(',')[2])
    size = float(message.split(' @')[3].split(',')[3])
    j = message.split(' @')[4].split('_')[0]
    i = message.split(' @')[4].split('_')[1]
    imageName = path.split('\\')[-1].split('/')[-1].split('.')[
        0] +'@' + j + '_' + i + '@'  + str(y_shift) + '_' + str(x_shift) + '_' + \
               str(angle) + '_' + str(size)
    out_name = img_Dir + '/' + imageName
    if os.path.exists(out_name + '.tif'):
        pass
    else:
        im_proj, im_geotrans, im_data = ReadTifImg(path)
        if angle % 90 == 0:
            y_ = [y - int(h * size / 2) + int(y_shift * size), y + int(h * size) - int(h * size / 2) +
                  int(y_shift * size)]
            x_ = [x - int(w * size / 2) + int(x_shift * size), x + int(w * size) - int(w * size / 2) +
                  int(x_shift * size)]
            if len(im_data.shape) == 3:
                im_data = im_data[:, y_[0]:y_[1], x_[0]:x_[1]]
                im_data = np.rot90(im_data, k=int((angle % 360) / 90), axes=(1, 2))
                im_data = im_data.transpose(1, 2, 0)
                im_data = cv2.resize(im_data, dsize=(w, h), interpolation=cv2.INTER_LINEAR)
                im_data = im_data.transpose(2, 0, 1)
            else:
                im_data = im_data[y_[0]:y_[1], x_[0]:x_[1]]
                im_data = np.rot90(im_data, k=int((angle % 360) / 90), axes=(0, 1))
                im_data = cv2.resize(im_data, dsize=(w, h), interpolation=cv2.INTER_LINEAR)
            im_geotrans = SetCutting(im_geotrans, x_[0], y_[0])
            im_geotrans = SetSize(im_geotrans, size)
            # time4 = datetime.datetime.now()
        else:
            im_proj, im_geotrans, im_data = area_cutting(im_proj, im_geotrans, im_data, h, w, x + x_shift,
                                                         y + y_shift, angle, size)
        if mode == 'w':
            WriteTifImg(out_name + '.tif', im_proj, im_geotrans, im_data)
            return out_name + '.tif', im_proj, im_geotrans, im_data
        elif mode == 'r':
            return im_proj, im_geotrans, im_data,imageName
        else:
            pass

def message_to_img_v2_with_pad(message='', data={}, img_Dir='', mode='r'):
    '''
    校验者：wyc 2022.02.23
    功能：
    :param message:
    :param data:no use
    :param img_Dir:
    :param mode:
    :return:
    '''
    # message = src_path + ' @' + 'y,x' + ' @' + 'h,w' + ' @' + 'y_shift,x_shift,angle,size' + ' @' + 'j,i'
    # time1 = datetime.datetime.now()
    path = message.split(' @')[0]
    y = int(message.split(' @')[1].split(',')[0])
    x = int(message.split(' @')[1].split(',')[1])
    h = int(message.split(' @')[2].split(',')[0])
    w = int(message.split(' @')[2].split(',')[1])
    y_shift = int(message.split(' @')[3].split(',')[0])
    x_shift = int(message.split(' @')[3].split(',')[1])
    angle = int(message.split(' @')[3].split(',')[2])
    size = float(message.split(' @')[3].split(',')[3])
    j = message.split(' @')[4].split('_')[0]
    i = message.split(' @')[4].split('_')[1]
    imageName = path.split('\\')[-1].split('/')[-1].split('.')[
        0] +'@' + j + '_' + i + '@'  + str(y_shift) + '_' + str(x_shift) + '_' + \
               str(angle) + '_' + str(size)
    out_name = img_Dir + '/' + imageName
    if os.path.exists(out_name + '.tif'):
        pass
    else:
        im_proj, im_geotrans, im_data = ReadTifImg(path)
        if angle % 90 == 0:
            y_ = [y - int(h * size / 2) + int(y_shift * size), y + int(h * size) - int(h * size / 2) +
                  int(y_shift * size)]
            x_ = [x - int(w * size / 2) + int(x_shift * size), x + int(w * size) - int(w * size / 2) +
                  int(x_shift * size)]
            if len(im_data.shape) == 3:
                im_data = im_data[:, y_[0]:y_[1], x_[0]:x_[1]]
                im_data = np.rot90(im_data, k=int((angle % 360) / 90), axes=(1, 2))
                # im_data = im_data.transpose(1, 2, 0)
                # im_data = cv2.resize(im_data, dsize=(w, h), interpolation=cv2.INTER_LINEAR)
                # im_data = im_data.transpose(2, 0, 1)
            else:

                im_data = im_data[y_[0]:y_[1], x_[0]:x_[1]]
                im_data = np.rot90(im_data, k=int((angle % 360) / 90), axes=(0, 1))
                # im_data = cv2.resize(im_data, dsize=(w, h), interpolation=cv2.INTER_LINEAR)
            im_geotrans = SetCutting(im_geotrans, x_[0], y_[0])
            im_geotrans = SetSize(im_geotrans, size)
            # time4 = datetime.datetime.now()
        else:
            # if
            im_proj, im_geotrans, im_data = area_cutting(im_proj, im_geotrans, im_data, h, w, x + x_shift,
                                                         y + y_shift, angle, size)
        if mode == 'w':
            WriteTifImg(out_name + '.tif', im_proj, im_geotrans, im_data)
            return out_name + '.tif', im_proj, im_geotrans, im_data
        elif mode == 'r':
            return im_proj, im_geotrans, im_data,imageName
        else:
            pass

def message_to_img_v2_with_data(message='', data={}, img_Dir='', mode='r'):
    '''
    校验者：wyc 2022.02.23
    功能：
    :param message:
    :param data:         im_proj = data[path]['proj']
                         im_geotrans = data[path]['geotrans']
                         im_data = data[path]['data']
    :param img_Dir:
    :param mode:
    :return:
    '''
    # message = src_path + ' @' + 'y,x' + ' @' + 'h,w' + ' @' + 'y_shift,x_shift,angle,size' + ' @' + 'j,i'
    # time1 = datetime.datetime.now()
    path = message.split(' @')[0]
    y = int(message.split(' @')[1].split(',')[0])
    x = int(message.split(' @')[1].split(',')[1])
    h = int(message.split(' @')[2].split(',')[0])
    w = int(message.split(' @')[2].split(',')[1])
    y_shift = int(message.split(' @')[3].split(',')[0])
    x_shift = int(message.split(' @')[3].split(',')[1])
    angle = int(message.split(' @')[3].split(',')[2])
    size = float(message.split(' @')[3].split(',')[3])
    j = message.split(' @')[4].split('_')[0]
    i = message.split(' @')[4].split('_')[1]
    imageName = path.split('\\')[-1].split('/')[-1].split('.')[
        0] +'@' + j + '_' + i + '@'  + str(y_shift) + '_' + str(x_shift) + '_' + \
               str(angle) + '_' + str(size)
    out_name = img_Dir + '/' + imageName
    if os.path.exists(out_name + '.tif'):
        pass
    else:
        im_proj = data[path]['proj']
        im_geotrans = data[path]['geotrans']
        im_data = data[path]['data']

        if angle % 90 == 0:
            y_ = [y - int(h * size / 2) + int(y_shift * size), y + int(h * size) - int(h * size / 2) +
                  int(y_shift * size)]
            x_ = [x - int(w * size / 2) + int(x_shift * size), x + int(w * size) - int(w * size / 2) +
                  int(x_shift * size)]
            if len(im_data.shape) == 3:
                im_data = im_data[:, y_[0]:y_[1], x_[0]:x_[1]]
                im_data = np.rot90(im_data, k=int((angle % 360) / 90), axes=(1, 2))
                im_data = im_data.transpose(1, 2, 0)
                im_data = cv2.resize(im_data, dsize=(w, h), interpolation=cv2.INTER_LINEAR)
                im_data = im_data.transpose(2, 0, 1)
            else:
                im_data = im_data[y_[0]:y_[1], x_[0]:x_[1]]
                im_data = np.rot90(im_data, k=int((angle % 360) / 90), axes=(0, 1))
                im_data = cv2.resize(im_data, dsize=(w, h), interpolation=cv2.INTER_LINEAR)
            im_geotrans = SetCutting(im_geotrans, x_[0], y_[0])
            im_geotrans = SetSize(im_geotrans, size)
            # time4 = datetime.datetime.now()
        else:
            im_proj, im_geotrans, im_data = area_cutting(im_proj, im_geotrans, im_data, h, w, x + x_shift,
                                                         y + y_shift, angle, size)
        if mode == 'w':
            WriteTifImg(out_name + '.tif', im_proj, im_geotrans, im_data)
            return out_name + '.tif', im_proj, im_geotrans, im_data
        elif mode == 'r':
            return im_proj, im_geotrans, im_data,imageName
        else:
            pass

from tqdm import trange
# 读取相应的图像data
def data_make(mess_list,data = {}):
    # data = {}
    for line in mess_list:
        data.update({line.split(' @')[0]:{}})
    with trange(len(list(data.keys()))) as t:
         for path,index in zip(list(data.keys()),t): #/media/gis/新加卷/数据仓库
            try:
                im_proj, im_geotrans, im_data = ReadTifImg(path)
                data.update({'%s' % path: {'proj': im_proj, 'geotrans': im_geotrans, 'data': im_data}})
                # print('data load sucessfully: %s'%path)
            except:
                print('fail read %s'%path)
        # del im_proj,im_geotrans,im_data
    return data

import torch
import random
from PIL import Image, ImageOps, ImageFilter

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        try:
            for keys_ in sample.keys():
                if keys_ == 'id': continue
                if keys_ == 'im_geotrans': continue
                if keys_ == 'im_proj': continue
                im_data = np.array(sample[keys_])
                if len(im_data.shape)==3:
                    sample[keys_] = im_data.astype(np.float32).transpose((2, 0, 1))
                else:
                    im_data = np.expand_dims(im_data,0)
                    sample[keys_] = im_data.astype(np.float32)
                sample[keys_] = torch.from_numpy(sample[keys_]).float()
        except:
            print('error','ToTensor',sample['id'])
            exit(0)
        return sample


class RandomHorizontalFlip(object):
    def __call__(self, sample):
        if random.random() < 0.5:
            try:
                for keys_ in sample.keys():
                    if keys_ == 'id': continue
                    if keys_ == 'im_geotrans': continue
                    if keys_ == 'im_proj': continue
                    sample[keys_] = sample[keys_].transpose(Image.FLIP_LEFT_RIGHT)
            except:
                print('error', sample['id'])
        if random.random() < 0.5:
            try:
                for keys_ in sample.keys():
                    if keys_ == 'id': continue
                    if keys_ == 'im_geotrans': continue
                    if keys_ == 'im_proj': continue
                    sample[keys_] = sample[keys_].transpose(Image.FLIP_TOP_BOTTOM)
            except:
                print('error', sample['id'])
        return sample



class RandomRotate(object):
    def __init__(self, degree):
        self.degree = degree
    def __call__(self, sample):
        rotate_degree = random.uniform(-1 * self.degree, self.degree)
        try:
            for keys_ in sample.keys():
                if keys_ == 'id': continue
                if keys_ == 'im_geotrans': continue
                if keys_ == 'im_proj': continue
                sample[keys_] = sample[keys_].rotate(rotate_degree, Image.BILINEAR) if keys_ == 'image' else \
                                sample[keys_].rotate(rotate_degree, Image.NEAREST)
        except:
            print('error',sample['id'])
        return sample
