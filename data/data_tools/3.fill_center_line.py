import os
import sys

import cv2
from tqdm import trange
from osgeo import gdal
import numpy as np

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

if __name__ == '__main__':

    root_dir = r"../train/SpaceNet"
    expand_num = 24 #扩充像素
    kernel = np.ones((expand_num, expand_num), np.uint8)

    for _file in os.listdir(root_dir):
        city_file = root_dir.replace("\\",'/') + '/' + _file
        center_line_file = city_file + '/' + 'label_center_line'
        out_roads_file = city_file + '/' + 'label_width%s'%expand_num
        os.makedirs(out_roads_file,exist_ok=True)
        with trange(len(os.listdir(center_line_file))) as t:
            for __file,index in zip(os.listdir(center_line_file),t):
                if os.path.exists(out_roads_file + '/' + __file):
                    continue
                im_proj, im_geotrans, im_data = ReadTifImg(center_line_file + '/' + __file)
                img_dilate = cv2.dilate(im_data, kernel, iterations=1)
                __file = __file.replace('.tif', '_mask.tif')
                WriteTifImg(out_roads_file + '/' + __file, im_proj, im_geotrans, img_dilate)




