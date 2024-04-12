# -*-coding:utf-8-*-
# !/usr/bin/env python
# @Time    : 2023/3/22 上午10:41
# @Author  : 王玉川 uestc
# @File    : spaceNet_loader.py
# @Description :
# if __name__ == '__main__':
#     __name__ = 'A7_GAMSNet.model.abc'
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Created by: Wang Yuchuan
## School of Automation Engineering, University of Electronic Science and Technology of China
## Email: wang-yu-chuan@foxmail.com
## Copyright (c) 2017
## for paper: A Multiscale and Multidirection Feature Fusion Network for Road Detection From Satellite Imagery
# https://ieeexplore.ieee.org/document/10477437)
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
from PIL import Image, ImageOps, ImageFilter

from sklearn.model_selection import train_test_split
import json
import torch.utils.data as data
import random

import sys,os
sys.path.append(os.path.abspath((os.path.dirname(__file__))))
try:
   from .data_process import *
except:
   from data_process import *
print('cwd:',os.getcwd())
cfg_dir = str(os.getcwd())+'/cfg_file/cfg_link_to_data_api.json'
def create_train_list(root_dir,train_sample_path,train_size,random_state):
    print('data use:',train_sample_path)
    os.makedirs((train_sample_path),exist_ok=True)
    if len(os.listdir(train_sample_path)) == 0:
        ################### data loader ################################# 981 + 257 + 1028 + 283
        img_dir = lab_dir = root_dir
        image_srcList = getDatapath(img_dir, dir_command='*/*/*', key_word='_sat')
        image_labList = getDatapath(lab_dir, dir_command='*/*/*', key_word='_mask')
        image_srcList = key_sort_for_SpaceNet(image_srcList)
        image_labList = key_sort_for_SpaceNet(image_labList)

        image_data_list = ['%s @%s' % (mess1, mess2) for mess1, mess2 in zip(image_srcList, image_labList)]
        data_check_for_SpaceNet(image_data_list)  # 这行代码具体的校验方式，还应该根据具体的实际情况而定
        trainlist, \
        vallist = train_test_split(image_data_list, train_size=train_size, random_state=random_state)
        with open(train_sample_path + 'train.txt', 'w+', encoding='utf-8') as f:
            for mess_ in trainlist:
                f.writelines(mess_ + '\n')
        with open(train_sample_path + 'val.txt', 'w+', encoding='utf-8') as f:
            for mess_ in vallist:
                f.writelines(mess_ + '\n')
        print('训练集数量：%d，验证集数量%d' % (len(trainlist), len(vallist)))
        # DIR_SETTING["train_sample"] = 'dataset_split_text/sample_split_dir'
    else:
        # DIR_SETTING["train_sample"] = 'dataset_split_text/sample_split_dir'
        print('本地样本划分！！！')

############################################加载配置#####################################################################

class CFG(object):
    def __init__(self):
        with open(cfg_dir, mode='r', encoding='utf-8') as f:
            cfg_list = json.load(f)
        self.DIR_SETTING = cfg_list[0]['DIR_SETTING']
        self.DATA_SETTING = cfg_list[1]['DATA_SETTING']
        self.OPTIMIZER_SETTING = cfg_list[2]['OPTIMIZER_SETTING']
        self.TRAIN_SETTING = cfg_list[3]['TRAIN_SETTING']
        self.TEST_SETTING = cfg_list[4]['TEST_SETTING']
        self.BEI_ZHU = cfg_list[5]['BEI_ZHU']
if os.path.exists(cfg_dir):
   cfg_data_api = CFG()
else:
    print('未找到配置文件，如果是训练阶段随后会出现严重bug')
########################################################################################################################

def SpaceNet_data_512_512(src_img, lab_img, imageName, test_flage=False):
    size_h = 512
    size_w = 512
    if test_flage==False:
        src_img = src_img[:,10:-10,10:-10]
        lab_img = lab_img[10:-10,10:-10]
    else:
        size_h=650
        size_w=650
    # 随机剪切
    center_x = np.array([256, 512, 768, 1024])
    center_y = np.array([256, 512, 768, 1024])
    angle_ = cfg_data_api.TRAIN_SETTING['ROTATE']
    if  test_flage == False:
        rand_x = random.choice(center_x)
        rand_y = random.choice(center_y)
        rand_angle = random.choice(angle_)
    else:
        rand_x = 650
        rand_y = 650
        rand_angle = 0
    img = src_img[:,rand_x-size_h//2:rand_x+size_h//2,rand_y-size_w//2:rand_y+size_w//2]
    label = lab_img[rand_x-size_h//2:rand_x+size_h//2,rand_y-size_w//2:rand_y+size_w//2]

    img1 = Image.fromarray(img.transpose(1, 2, 0))
    mask1 = Image.fromarray(label)
    # 翻折+旋转
    if random.random() < 0.5 and test_flage == False:
        img1 = img1.transpose(Image.FLIP_LEFT_RIGHT)
        mask1 = mask1.transpose(Image.FLIP_LEFT_RIGHT)
    if random.random() < 0.5 and test_flage == False:
        img1 = img1.rotate(rand_angle, Image.BILINEAR)
        mask1 = mask1.rotate(rand_angle, Image.BILINEAR)
    imageName_ = imageName + '(%d_%d_%d)' % (rand_x, rand_y, rand_angle)

    return  img1, mask1, imageName_


########################################################################################################################
class SpaceNet_dataloader():
    def __init__(self):
        if cfg_data_api.DATA_SETTING['deep_lodader_type'] == '512_512':
            self.deepGlobe_data_load_Type = SpaceNet_data_512_512
        else:
            raise Exception("TRAIN_SETTING['deep_lodader_type'] 设置异常，请检查！！")
            exit(0)

    def _loader_deepGlobe(self, data_list, index, data_dict=[None], class_num=1, input_normalization=False,
                          DataEnchance=True, mixType=False, u=0.5, test_flage=False):  # 这里只导入了原始数据的，用的就是这个
        # 数据加载
        img, mask, imageName = message_to_img_v1(data_list[index])
        try:
            if mask.sum() < 1024 * 255*3: #去除大面积没有道路的标签
                blank_mess = '%s %d' % (data_list[index].split(' @')[0], mask.sum())
                index_choose = np.random.choice(range(0, len(list(data_list))))
                img, mask, imageName = message_to_img_v1(data_list[index_choose])
                blank_mess = blank_mess + '\n%s %d\n\n' % (data_list[index_choose].split(' @')[0], mask.sum())
                with open('./blank_data.log', 'a+') as f:
                    f.write(blank_mess)
                print('waring code 303',data_list[index], data_list[index_choose], '\n')
        except:
            print('waring code 403',data_list[index])
        try:
            img, mask, imageName = self.deepGlobe_data_load_Type(img, mask,imageName, test_flage)
        except:
            print(imageName)
            exit(0)

        img = np.array(img).transpose([2,0,1])
        mask = np.array(mask)
        ####################################################################################################################
        mask_s = np.zeros(mask.shape,dtype=np.float32)

        # 转化形状并标准化
        if input_normalization:
            self.mean = (0.485, 0.456, 0.406)
            self.std = (0.229, 0.224, 0.225)
            img /= 255.0
            img -= self.mean
            img /= self.std
        else:
            img = np.array(img, np.float32)

        if class_num == 1:  # 多分类标签数据的处理
            mask_s[mask >= 0.5] = 1
            mask_s[mask < 0.5] = 0
        else:  # 灵活更改 输入标签的类型
            # mask[mask == 0] = 0   # beijing
            mask[mask != (80 or 160 or 240)] = 0  # beijing 可能会出错
            mask[mask == 80] = 1  # road
            mask[mask == 160] = 2  # building
            mask[mask == 240] = 0  # river

        mask_s = np.expand_dims(mask_s,axis=0)
        if not class_num == 1:
            mask_s = get_one_hot(mask_s,class_num)


        return img, mask_s, imageName

class ImageFolder(data.Dataset):

    def __init__(self, train_list, readType='v1', class_num=1,
                 multi_scale_pred = False,
                 input_normalization=False,
                 DataEnchance=False, mixType=False,
                 u=0.5, test_flage=False):
        self.input_normalization = input_normalization
        self.DataEnchance = DataEnchance
        self.mixType = mixType

        # train_list = args.data_list[split]
        self.data_list = train_list
        self.loader = SpaceNet_dataloader()._loader_deepGlobe
        self.get_one_hot = get_one_hot
        self.u = u
        self.readType = readType
        self.class_num = class_num
        self.test_flage = test_flage

        self.multi_scale_pred = multi_scale_pred
        self.angle_theta = 10

        if readType == 'v1':
            self.src_data = None
            self.label_data = None

    def __getitem__(self, index):
        # get road data
        image, mask, imageName = self.loader(self.data_list, index,
                                                                 [self.src_data, self.label_data],
                                                                 class_num=self.class_num,
                                                                 input_normalization=self.input_normalization,
                                                                 DataEnchance=self.DataEnchance, mixType=self.mixType,
                                                                 u=self.u, test_flage=self.test_flage)
        mask = torch.Tensor(mask)
        image = torch.Tensor(image)
        return {'image': image,
                'label': mask,
                 'id': imageName}

    def __len__(self):
        return len(list(self.data_list))

def loader_get(train_sample_path,opt={}):
    trainlist = []
    vallist = []
    with open(train_sample_path + '/train.txt', 'r', encoding='utf-8') as f:
        for mess_ in f.readlines():
            trainlist.append(mess_.replace('\n', ''))
    with open(train_sample_path + '/val.txt', 'r', encoding='utf-8') as f:
        for mess_ in f.readlines():
            vallist.append(mess_.replace('\n', ''))

    train_dataset = ImageFolder(trainlist,
                                class_num = cfg_data_api.TRAIN_SETTING['CLASS_NUM'],
                                multi_scale_pred=cfg_data_api.DATA_SETTING['multi_scale_pred'])
    val_dataset = ImageFolder(vallist,
                              class_num=cfg_data_api.TRAIN_SETTING['CLASS_NUM'],
                              multi_scale_pred=cfg_data_api.DATA_SETTING['multi_scale_pred'],test_flage=True)

    data_loader = data.DataLoader(
        train_dataset,
        batch_size=opt["train_batchsize"],
        shuffle=opt["shuffle"],
        num_workers=opt["train_num_workers"],
        pin_memory=opt["train_pin_memor"])
    val_data_loader = data.DataLoader(
        val_dataset,
        batch_size=opt["val_batchsize"],
        shuffle=False,
        num_workers=opt["val_num_workers"],
        pin_memory=opt["val_pin_memor"])
    return data_loader,val_data_loader


