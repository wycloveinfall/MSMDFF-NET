# -*-coding:utf-8-*-
# !/usr/bin/env python2

"""
create_png.py: script to convert Spacenet 11-bit RGB images to 8-bit images and
preprocessing with CLAHE (a variant of adaptive histogram equalization algorithm).

It will create following directory structure:
    base_dir
        | ---> RGB_8bit : Save 8-bit png images.
"""

from __future__ import print_function

import os
import numpy as np
import cv2
from tqdm import tqdm

tqdm.monitor_interval = 0

from osgeo import gdal
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

    # GeoTransform[0]  横向 水平
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


def Create_single_tif(input_data):
    # 直方图均衡，
    clahe = cv2.createCLAHE(clipLimit=2, tileGridSize=(8, 8))

    red = np.asarray(input_data[0], dtype=np.float32)
    green = np.asarray(input_data[1], dtype=np.float32)
    blue = np.asarray(input_data[2], dtype=np.float32)

    red_ = 255.0 * ((red - np.min(red)) / (np.max(red) - np.min(red)))
    green_ = 255.0 * ((green - np.min(green)) / (np.max(green) - np.min(green)))
    blue_ = 255.0 * ((blue - np.min(blue)) / (np.max(blue) - np.min(blue)))

    ## The default image size of Spacenet Dataset is 1300x1300.
    img_rgb = np.zeros((3, 1300, 1300), dtype=np.uint8)
    img_rgb[0] = clahe.apply(np.asarray(red_, dtype=np.uint8))
    img_rgb[1] = clahe.apply(np.asarray(green_, dtype=np.uint8))
    img_rgb[2] = clahe.apply(np.asarray(blue_, dtype=np.uint8))

    return img_rgb

from tqdm import trange
def main():
    print('cwd',os.getcwd())
    root_dir = os.getcwd()+"/data/train/SpaceNet"
    for _file in os.listdir(root_dir):
        city_file = root_dir.replace("\\",'/') + '/' + _file
        # geojson_roads_file = city_file + '/' + 'geojson_roads'
        RGB_roads_file = city_file + '/' + 'PS-RGB'
        out_roads_file = city_file + '/' + 'RGB-8bit'
        os.makedirs(out_roads_file,exist_ok=True)
        with trange(len(os.listdir(RGB_roads_file))) as t:
            for __file,index in zip(os.listdir(RGB_roads_file),t):
                im_proj, im_geotrans, im_data = ReadTifImg(RGB_roads_file + '/' + __file)
                out_RGB8bit = Create_single_tif(im_data)
                __file = __file.replace('.tif','_sat.tif')
                WriteTifImg(out_roads_file + '/' + __file, im_proj, im_geotrans, out_RGB8bit)




if __name__ == "__main__":
    main()