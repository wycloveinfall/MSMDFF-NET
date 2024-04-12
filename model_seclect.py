# -*-coding:utf-8-*-
# !/usr/bin/env python
# @Time    : 2023/3/8 下午3:31
# @Author  : 王玉川 uestc
# @File    : model_seclect.py
# @Description :

import json

# UNet 系列
# from A1_UNetpp import UNetPP

from networks.A2_DeepRoadMapper import DeepRoadMapper_segment

from networks.A3_LinkNet_series import LinkNet18,LinkNet34,LinkNet50,LinkNet101

from networks.A4_DlinkNet_series import DinkNet18,DinkNet34,DinkNet50,DinkNet101,DinkNet34_less_pool

from networks.A5_RoadCNN import RoadCNN

from networks.A7_GAMSNet.GAMS_Net import GAMSNet

from networks.A8_CoANet.coanet_without_HB import CoANet

from networks.A9_CADUNet.CAD_UNet import Cad_UNet

from networks.A10_SGCNNet.SGCNNet import SGCN_res50

from networks.A11_MSMDFFNet.model.MSMDFF_Net import MSMDFF_Net_v3_plus

#### 消融实验
from networks.A11_MSMDFFNet.model.LinkNet34_MDFF import LinkNet34_MDFF
from networks.A11_MSMDFFNet.model.LinkNet34_MDCF import LinkNet34_MDCF_s2

from networks.A11_MSMDFFNet.model.LinkNet34_MDFF_MDCF import LinkNet34_MDFF_MDCF_v2
from networks.A11_MSMDFFNet.model.LinkNet34_MDFF_MSR import LinkNet34_MDFF_MSR
from networks.A11_MSMDFFNet.model.LinkNet34_MSR_MDCF import LinkNet34_MSR_MDCF


from networks.A11_MSMDFFNet.model.LinkNet34_MDFF import LinkNet34_MDFF_strip_size_3
from networks.A11_MSMDFFNet.model.LinkNet34_MDFF import LinkNet34_MDFF_strip_size_5
from networks.A11_MSMDFFNet.model.LinkNet34_MDFF import LinkNet34_MDFF_strip_size_7
from networks.A11_MSMDFFNet.model.LinkNet34_MDFF import LinkNet34_MDFF_strip_size_9
from networks.A11_MSMDFFNet.model.LinkNet34_MDFF import LinkNet34_MDFF_strip_size_11
from networks.A11_MSMDFFNet.model.LinkNet34_MDFF import LinkNet34_MDFF_strip_size_13

from networks.A11_MSMDFFNet.model.LinkNet34_MDCF import LinkNet34_MDCF_s2_strip_size_3
from networks.A11_MSMDFFNet.model.LinkNet34_MDCF import LinkNet34_MDCF_s2_strip_size_5
from networks.A11_MSMDFFNet.model.LinkNet34_MDCF import LinkNet34_MDCF_s2_strip_size_7
from networks.A11_MSMDFFNet.model.LinkNet34_MDCF import LinkNet34_MDCF_s2_strip_size_9
from networks.A11_MSMDFFNet.model.LinkNet34_MDCF import LinkNet34_MDCF_s2_strip_size_11
from networks.A11_MSMDFFNet.model.LinkNet34_MDCF import LinkNet34_MDCF_s2_strip_size_13


from networks.A12_MSMDFFvsCoANet.model.MSMDFF_Net import MSMDFF_Net_base
from networks.A12_MSMDFFvsCoANet.model.coanet_without_HB import CoANet_base


class Model_get(object):
    '''
    author:wyc
    2023.11.21 原来的代码中获取的net需要再加一个()这样的话就完成 __call__()调用
    '''
    def __init__(self,Model_select=None,cfg_path = './cfg_file/cfg_link_to_data_api.json',device_choose = 'GPU'):
        self.cfg_path = cfg_path
        self.device_choose = device_choose

        try:
            with open(cfg_path, mode='r', encoding='utf-8') as f:
                cfg_list = json.load(f)
            TRAIN_SETTING = cfg_list[3]['TRAIN_SETTING']
            Model_select = TRAIN_SETTING['MODEL_TYPE'] if Model_select==None else Model_select
            TRAIN_IMG_SIZE = TRAIN_SETTING['TRAIN_IMG_SIZE']
            class_num = TRAIN_SETTING['CLASS_NUM']
            self.DIR_SETTING = cfg_list[0]['DIR_SETTING']
            self.CONFIG_NAME = cfg_list[6]['CONFIG_NAME']

        except:
            # Model_select = cfg_path
            class_num = 1
            TRAIN_IMG_SIZE = [3,512,512]

        self.TRAIN_IMG_SIZE = TRAIN_IMG_SIZE
        self.class_num = class_num
        self.Model_select = Model_select

    def __call__(self):
        Model_select = self.Model_select
        device_choose = self.device_choose
        TRAIN_IMG_SIZE = self.TRAIN_IMG_SIZE
        class_num = self.class_num

        ############################################# Unet 系列 ##########################################################
        if Model_select == "UNet":# todo
            model = UNet(input_channels=TRAIN_IMG_SIZE[0], num_classes=class_num)
        elif Model_select == "UNetPP" or Model_select == "UNet++":
            model = UNetPP(input_channels=TRAIN_IMG_SIZE[0], num_classes=class_num)
        ##########################################  DeepRoadMapper系列 ####################################################

        elif Model_select == "DeepRoadMapper_segment" or Model_select == 'DeepRoadMapper':
            model = DeepRoadMapper_segment(num_classes=class_num)
        ##########################################  LinkNet 系列 ####################################################
        elif Model_select == "LinkNet18" or Model_select == "link18":#todo
            model = LinkNet18(num_classes=class_num)
        elif Model_select == "LinkNet34" or Model_select == "Link34":#todo
            model = LinkNet34( num_classes=class_num)
        elif Model_select == "LinkNet50" or Model_select == "Link50":#todo
            model = LinkNet50( num_classes=class_num)
        elif Model_select == "LinkNet101" or Model_select == "Link101":#todo
            model = LinkNet101( num_classes=class_num)
        ##########################################  DlinkNet 系列 ####################################################
        elif Model_select == "DinkNet18" or Model_select == "Dlink18" or Model_select == "DlinkNet18":#todo
            model = DinkNet18( num_classes=class_num)
        elif Model_select == "DinkNet34" or Model_select == "DlinkNet34":#todo
            model = DinkNet34( num_classes=class_num)
        elif Model_select == "DinkNet50" or Model_select == "DlinkNet50":
            model = DinkNet50( num_classes=class_num)
        elif Model_select == "DinkNet101"or Model_select == "DlinkNet101":
            model = DinkNet101( num_classes=class_num)
        elif Model_select == "DinkNet34_less_pool":
            model = DinkNet34_less_pool( num_classes=class_num)
        ##########################################  RoadCNN 系列 ####################################################
        elif Model_select == "RoadCNN":
            model = RoadCNN(num_classes=class_num)

        ########################################### GAMSNet_RT ####################################################
        elif Model_select == "GAMSNet_RT":#
            model = GAMSNet(replicated='repeat',grad_repeat=True)
        elif Model_select == "GAMSNet_RF":#
            model = GAMSNet(replicated='repeat',grad_repeat=False)
        elif Model_select == "GAMSNet_TT":#
            model = GAMSNet(replicated='title',grad_repeat=True)
        elif Model_select == "GAMSNet_TF":#
            model = GAMSNet(replicated='title',grad_repeat=False)
        ############################################################################################################

        elif Model_select == "SGCNNet" or Model_select == "SGCN":
            model = SGCN_res50(num_classes=class_num)

        elif Model_select == "CoANet[withoutUB]" or Model_select == "CoANet":
            model = CoANet(num_classes=1,backbone='resnet',output_stride=8,sync_bn=False,freeze_bn=False)
        elif Model_select == "Cad_UNet" or Model_select == "CadUNet" or Model_select == "CADUNet":
            model = Cad_UNet(num_classes=class_num)
        ##########################################  MSMDFF Net系列 ####################################################

        elif Model_select == "LinkNet34_MDCF_s2": # 使用旋转特征图的方式
            model = LinkNet34_MDCF_s2(num_classes=class_num)

        elif Model_select == "LinkNet34_MDFF":
            model = LinkNet34_MDFF(num_classes=class_num)

        elif Model_select == "MSMDFF_Net_v3_plus":
            model = MSMDFF_Net_v3_plus(num_classes=class_num)


        elif Model_select == "LinkNet34_MDFF_MDCF_v2":
            model = LinkNet34_MDFF_MDCF_v2(num_classes=class_num)
        elif Model_select == "LinkNet34_MDFF_MSR":
            model = LinkNet34_MDFF_MSR(num_classes=class_num)
        elif Model_select == "LinkNet34_MSR_MDCF":
            model = LinkNet34_MSR_MDCF(num_classes=class_num)

        # 超参数的调整 实验 initblock中带状卷积的超参数

        elif Model_select == "LinkNet34_MDFF_strip_size_3":
            model = LinkNet34_MDFF_strip_size_3(num_classes=class_num)
        elif Model_select == "LinkNet34_MDFF_strip_size_5":
            model = LinkNet34_MDFF_strip_size_5(num_classes=class_num)
        elif Model_select == "LinkNet34_MDFF_strip_size_7":
            model = LinkNet34_MDFF_strip_size_7(num_classes=class_num)
        elif Model_select == "LinkNet34_MDFF_strip_size_9":
            model = LinkNet34_MDFF_strip_size_9(num_classes=class_num)
        elif Model_select == "LinkNet34_MDFF_strip_size_11":
            model = LinkNet34_MDFF_strip_size_11(num_classes=class_num)
        elif Model_select == "LinkNet34_MDFF_strip_size_13":
            model = LinkNet34_MDFF_strip_size_13(num_classes=class_num)

        # 超参数调整 实验 解码器
        elif Model_select == "LinkNet34_MDCF_s2_strip_size_3":
            model = LinkNet34_MDCF_s2_strip_size_3(num_classes=class_num)
        elif Model_select == "LinkNet34_MDCF_s2_strip_size_5":
            model = LinkNet34_MDCF_s2_strip_size_5(num_classes=class_num)
        elif Model_select == "LinkNet34_MDCF_s2_strip_size_7":
            model = LinkNet34_MDCF_s2_strip_size_7(num_classes=class_num)
        elif Model_select == "LinkNet34_MDCF_s2_strip_size_9":
            model = LinkNet34_MDCF_s2_strip_size_9(num_classes=class_num)
        elif Model_select == "LinkNet34_MDCF_s2_strip_size_11":
            model = LinkNet34_MDCF_s2_strip_size_11(num_classes=class_num)
        elif Model_select == "LinkNet34_MDCF_s2_strip_size_13":
            model = LinkNet34_MDCF_s2_strip_size_13(num_classes=class_num)

        elif Model_select == 'MSMDFF_USE_SCMD':
             model = MSMDFF_Net_base(num_classes=class_num,init_block_SCMD=True,decoder_use_SCMD=True) #
        elif Model_select == 'MSMDFF_USE_SCM':
             model = MSMDFF_Net_base(num_classes=class_num,init_block_SCMD=True,decoder_use_SCMD=False) #
        elif Model_select == 'MSMDFF_USE_SCM_all':
             model = MSMDFF_Net_base(num_classes=class_num,init_block_SCMD=False,decoder_use_SCMD=False) #

        elif Model_select == 'CoANet_USE_SCMD':
             model = CoANet_base(num_classes=1, backbone='resnet', output_stride=8, sync_bn=False, freeze_bn=False,is_use_SCMD=True)
        elif Model_select == 'CoANet_USE_SCM':
             model = CoANet_base(num_classes=1, backbone='resnet', output_stride=8, sync_bn=False, freeze_bn=False,is_use_SCMD=False)
        else:
            model=None
            print('model choose error!->>',Model_select)
            exit(0)

        return model


if __name__ == '__main__':
    import torch
    net = Model_get('RE_DNNet',device_choose='CPU')
    example1 = torch.rand(1, 3, 512, 512).cpu()
    CNN = net()
    out = CNN(example1)
    print(CNN)