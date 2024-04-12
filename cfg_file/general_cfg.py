# coding=utf-8
import os,sys
import json
from pathlib import Path
sys.path.append(str(Path(os.getcwd())))
CONFIG_NAME = __file__.replace('\\', '/').split('/')[-1][:-3]  # 获取当前 运行的文件名字

BEI_ZHU = {'main':'default:soft iou + bce loss'}

test_metric_display = 1

################# you need to change following parameters ##############
DATA_TYPE_flage = '2' #todo you can set 0 1 2 to select SP DP MA dataset
batch_size = 1 #todo
NUMBER_WORKERS = 0 #todo
image_size = [3,512,512] #todo
SHUFFLE = True
MODEL_SELECT_flag = "17" #todo

LOSS_FUNCTION_flage = "0" #todo you can choose different loss function

LR_SCHEDULER_flage = "6" #todo you can choose different methods for updating learning rate

OPTIMIZER_flage = "6" #todo you can select different OPTIMIZER

train_model_savePath = '/home/wyc/mount/DISCK_1/train_data_cache' #todo choose your save path

######################### end ####################################

DATA_TYPE ={
    "-1":'v1',
    "0" : 'spacenet_512x512',
    "1" : 'DeepGlobe_512x512',
    "2" : 'Massachusetts_512x512'
}
_MODEL_SELECT = {
    ############## UNet 系列 ###############
    "1": "UNet",
    "2": "UNet++",
    ############## LinkNet 系列 #############
    "5": "LinkNet18",
    "6": "DinkNet18",
    "7": "LinkNet34",
    "8": "DinkNet34",
    "9": "LinkNet50",
    "10": "DinkNet50",
    ############# DeepRoadMapper 2017 ##########
    "11": "DeepRoadMapper_segment",
    ############## RoadCNN 2018 ################
    "12": "RoadCNN",
    ############## Batra_net 2019 ##############
    "13": "Batra_net",
    ############## GAMSNet 2021 ################
    "14": "GAMSNet",
    ############## CADUNet 2021 ################
    "15": "CADUNet",
    ################## CoANet ##################
    "16": "CoANet[withoutUB]",
    ############## SGCNNet 2022 ################
    "17": "SGCNNet",
    ############### MSMDFF 2024 ##############
    "18": "LinkNet34_MDFF",
    "19": "LinkNet34_MDCF_s2",
    #
    "20": "LinkNet34_MDFF_MDCF_v2",
    "21": "LinkNet34_MDFF_MSR",
    "22": "LinkNet34_MSR_MDCF",
    #
    "23": "LinkNet34_MDFF_strip_size_3",
    "24": "LinkNet34_MDFF_strip_size_5",
    "25": "LinkNet34_MDFF_strip_size_7",
    "26": "LinkNet34_MDFF_strip_size_9",
    "27": "LinkNet34_MDFF_strip_size_11",
    "28": "LinkNet34_MDFF_strip_size_13",
    #
    "29": "LinkNet34_MDCF_s2_strip_size_3",
    "30": "LinkNet34_MDCF_s2_strip_size_5",
    "31": "LinkNet34_MDCF_s2_strip_size_7",
    "32": "LinkNet34_MDCF_s2_strip_size_9",
    "33": "LinkNet34_MDCF_s2_strip_size_11",
    "34": "LinkNet34_MDCF_s2_strip_size_13",
    #
    "35": "CoANet_USE_SCMD",
    "36": "CoANet_USE_SCM",
    "37": "MSMDFF_USE_SCMD",
    "38": "MSMDFF_USE_SCM",

}


_LOSS_FUNCTION = {
    '0':'dice_bce_loss',
    '1':'MSE_loss',
    '2':'DiceLoss',
    '3':'MulticlassDiceLoss',
    '4':'SoftIoULoss',
    '5':'MSE_loss_edges',
    '6':'dice_bce_loss_edges'
}

_LR_SCHEDULER = {
    '0': "StepLR",# 每隔n个epoch时调整学习率，具体是用当前学习率乘以gamma即lr=lr∗gamma
    '1':"MultiStepLR",#milestones设定调整时刻数 gamma调整系数 如构建个list设置milestones=[50,125,180]，在第50次、125次、180次时分别调整学习率，具体是用当前学习率乘以gamma即lr=lr∗gamma ；
    '2':"ExponentialLR",
    '3':"CosineAnnealingLR",
    '4':"ReduceLRonPlateau",
    '5':"LambdaLR",
    '6':"POLY"
    }

_OPTIMIZER = {
    '0':'SGD',
    '1':'ASGD',
    '2':'Rprop',
    '3':'Adagrad',
    '4':'Adadelta',
    '5':'RMSprop',
    '6':'Adam(AMSGrad)'}
########################################################################################################################
DIR_SETTING = {
    "image_dir":'',
    "label_dir":'',
    "train_sample": '../dataset_split_text/',
    "trainProcessdata": train_model_savePath,
    "json_file_dir":os.getcwd()}

DATA_SETTING = {
    "deep_lodader_type":'512_512',#todo

    "train_size": 0.85,
    "use_blank": 0,
    "random_state": 8,

    "input_normalization": False,
    "trainDataEnchance": True,
    "valDataEnchance": False,
    "pretrained": False,
    "data_plus": False,
    "multi_scale_pred": False
}
OPTIMIZER_SETTING={
    "LOSS_FCT": _LOSS_FUNCTION[LOSS_FUNCTION_flage],
    "OPTIMIZER":_OPTIMIZER[OPTIMIZER_flage],
    "LR_SCHEDULER": _LR_SCHEDULER[LR_SCHEDULER_flage],
    "MOMENTUM": 0.9,
    "WEIGHT_DECAY": 0.0005,
    "LR_INIT": 5e-3,  # 初始学习率
    "LR_END": 1e-6,
    "WARMUP_EPOCHS": 2,
}

TRAIN_SETTING = {
    "runType": 'Train',
    "TRAIN_IMG_SIZE": image_size,  # 输入图片的大小
    "DATA_TYPE":DATA_TYPE[DATA_TYPE_flage],
    "CLASS_NUM": 1,
    "MODEL_TYPE": _MODEL_SELECT[MODEL_SELECT_flag],
    "BATCH_SIZE": batch_size,  # 每次训练输入的batch大小
    "TRAIN_PIN_MEMOR": True,
    "IOU_THRESHOLD_LOSS": 0.5,  # 分割的IOU阈值
    "EPOCHS": 200,  # 总的训练批次
    "NUMBER_WORKERS": NUMBER_WORKERS,  # dataloader 使用的线程数
    # enhance
    # 物理变化
    "MULTI_SCALE_TRAIN": True,  # 不同分辨率图像混合训练，动态的分辨率训练
    "ROTATE": [0,30,60,120,150],
    "hsv_T":True,
    "shuffle":SHUFFLE,
}

# test
TEST_SETTING = {
    "DISPLAY_FREQ":test_metric_display,
    "TEST_IMG_SIZE": image_size,
    "BATCH_SIZE": batch_size,
    "NUMBER_WORKERS": NUMBER_WORKERS,
    "TEST_PIN_MEMOR": True,
    "CONF_THRESH": 0.01,
    "NMS_THRESH": 0.5,
    "MULTI_SCALE_TEST": False,
    "FLIP_TEST": False
}
_CLASSMASK = {
    "background": (0, 0, 0),
    "road": (255, 255, 0),  # 黄色
    "building": (0, 255, 0)  # 绿色
}

# model
_MODEL = {"ANCHORS": [[(1.25, 1.625), (2.0, 3.75), (4.125, 2.875)],  # Anchors for small obj 52*52
                     [(1.875, 3.8125), (3.875, 2.8125), (3.6875, 7.4375)],  # Anchors for medium obj 26*26
                     [(3.625, 2.8125), (4.875, 6.1875), (11.65625, 10.1875)]],  # Anchors for big obj 13*13  输入为416*416
         "STRIDES": [8, 16, 32],
         "ANCHORS_PER_SCLAE": 3
         }

if DATA_TYPE[DATA_TYPE_flage] == 'spacenet_512x512':
    from utils.dataloader._slice_loader.spaceNet_loader import create_train_list
    DIR_SETTING['root_dir'] = str(Path(os.getcwd())) + '/data/train/SpaceNet'
    DIR_SETTING['train_sample'] = str(Path(os.getcwd())) + '/dataset_split_text/spaceNet/'
elif DATA_TYPE[DATA_TYPE_flage] == 'DeepGlobe_512x512':
    from utils.dataloader._slice_loader.deepGlobe_loader import create_train_list
    DIR_SETTING['root_dir'] = str(Path(os.getcwd())) + '/data/train/DeepGlobe'
    DIR_SETTING['train_sample'] = str(Path(os.getcwd())) + '/dataset_split_text/deepglobe/'
elif DATA_TYPE[DATA_TYPE_flage] == 'Massachusetts_512x512':
    from utils.dataloader._slice_loader.Massachusetts_loader import create_train_list
    DIR_SETTING['root_dir'] = str(Path(os.getcwd())) + '/data/train/MassachusettsRoads'
    DIR_SETTING['train_sample'] = str(Path(os.getcwd())) + '/dataset_split_text/Massachusetts/'
else:
    print('dataType choose error!')
    exit(0)

train_sample_path = DIR_SETTING["train_sample"]
os.makedirs((train_sample_path),exist_ok=True)
create_train_list(DIR_SETTING["root_dir"],train_sample_path,
                  DATA_SETTING["train_size"],DATA_SETTING["random_state"])

print(os.getcwd())
cfg_file_path = str(Path(os.getcwd()))
with open(cfg_file_path + '/cfg_file/cfg_link_to_data_api.json', 'w', encoding='utf-8') as f:
    dict_write = []
    dict_write.append({'DIR_SETTING': DIR_SETTING})
    dict_write.append({'DATA_SETTING': DATA_SETTING})
    dict_write.append({'OPTIMIZER_SETTING': OPTIMIZER_SETTING})
    dict_write.append({'TRAIN_SETTING': TRAIN_SETTING})
    dict_write.append({'TEST_SETTING': TEST_SETTING})
    dict_write.append({'BEI_ZHU': BEI_ZHU})
    dict_write.append({'CONFIG_NAME': CONFIG_NAME})
    js = json.dumps(dict_write)
    js = js.replace('},', '},\n')
    f.write(js)
if __name__ == '__main__':
    #json 读取的接口
    class cfg(object):
        def __init__(self):
            with open(cfg_file_path+'/cfg_file/cfg_link_to_data_api.json', mode='r', encoding='utf-8') as f:
                cfg_list = json.load(f)
            self.DIR_SETTING = cfg_list[0]['DIR_SETTING']
            self.DATA_SETTING = cfg_list[1]['DATA_SETTING']
            self.OPTIMIZER_SETTING = cfg_list[2]['OPTIMIZER_SETTING']
            self.TRAIN_SETTING = cfg_list[3]['TRAIN_SETTING']
            self.TEST_SETTING = cfg_list[4]['TEST_SETTING']
            self.BEI_ZHU = cfg_list[5]['BEI_ZHU']
            self.CONFIG_NAME = cfg_list[6]['CONFIG_NAME']

    cfg()