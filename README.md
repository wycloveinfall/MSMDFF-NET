## A Multiscale and Multidirection Feature Fusion Network for Road Detection From Satellite Imagery

This paper ([MSMDFF-Net](https://ieeexplore.ieee.org/document/10477437)) has been published in IEEE TGRS 2024.

This code is licensed for non-commerical research purpose only.

### Introduction

The completeness of road extraction is very important for road application. However, existing deep learning (DP) methods of extraction often generate fragmented results. The prime reason is that DP-based road extraction methods use square kernel convolution, which is challenging to learn long-range contextual relationships of roads. The road often produce fractures in the local interference area. Besides, the quality of extraction results will be subjected to the resolution of remote sensing (RS) image. Generally, an algorithm will produce worse fragmentation when the used data differ from the resolution of the training set. To address these issues, we propose a novel road extraction framework for RS images, named the multiscale and multidirection feature fusion network (MSMDFF-Net). This framework comprises three main components: the multidirectional feature fusion (MDFF) initial block, the multiscale residual (MSR) encoder, and the multidirectional combined fusion (MDCF) decoder. First, according to the road’s morphological characteristics, we develop a strip convolution module with a direction parameter (SCM-D). Then, to make the extracted result more complete, four SCM-D with different directions are used to MDFF-initial block and multidirectional combined fusion decoder (MDCF-decoder). Finally, we incorporate an additional branch into the residual network (ResNet) encoding module to build multiscale residual encoder (MSR-encoder) for improving the generalization of the model on different resolution RS image. Extensive experiments on three popular datasets with different resolution (Massachusetts, DeepGlobe, and SpaceNet datasets) show that the proposed MSMDFF-Net achieves new state-of-the-art results. The code will be available at https://github.com/wycloveinfall/MSMDFF-NET.
![fig1.png](Figure%2Ffig1.png)
![fig2.png](Figure%2Ffig2.png)
### Citations

If you are using the code/model provided here in a publication, please consider citing:

    @article{wang2024MSMDFF-Net,
    title={A Multiscale and Multidirection Feature Fusion Network for Road Detection From Satellite Imagery},
    author={Wang, Yuchuan and Tong, Ling and Luo, Shiyu and Xiao, Fanghong and Yang, Jiaxing},
    journal={IEEE Transactions on Geoscience and Remote Sensing},
    volume={62},
    pages={1-18},
    year={2024},
    publisher={IEEE},
    doi={10.1109/TGRS.2024.3379988}}
    }

### Requirements

The code is built with the following dependencies:
- Python ==3.8.16
- [PyTorch](https://pytorch.org/get-started/previous-versions/) ==1.8.1
- opencv-python==4.7.0.72
- gdal==3.0.2
- scikit-learn==1.2.2
- tqdm==4.65.0
- tensorboard==2.13.0
- six==1.16.0 
- torchsummary==1.5.1
```
conda create -n torch1.8.1
conda activate torch1.8.1
conda install pytorch==1.8.1 torchvision==0.9.1 torchaudio==0.8.1 cudatoolkit=10.2 -c pytorch
conda install six==1.16.0
conda install gdal==3.0.2
pip install opencv-python==4.7.0.72
pip install scikit-learn==1.2.2
pip install tqdm==4.65.0
pip install tensorboard==2.13.0
pip install torchsummary==1.5.1
```

### Data Preparation
#### PreProcess SpaceNet Dataset
- Convert SpaceNet 11-bit images to 8-bit Images. 
```
cd MSMDFF-NET-main
python data/data_tools/SP_datatools/1.convert_to_8bit.py
```
- Create road centerline.
```
cd MSMDFF-NET-main
python data/data_tools/SP_datatools/2.cerete_label.py
```
- Create road masks.
```
cd MSMDFF-NET-main
python data/data_tools/SP_datatools/3.fill_center_line.py 
```
- select Train files. 
  Not all JSON files can generate masks, so it's necessary to filter out paired training samples.
```
cd MSMDFF-NET-main
python data/data_tools/SP_datatools/4.choose_train_flie.py 
```
*SpaceNet dataset tree structure.*

before preprocessing
```
spacenet
|
└───AOI_2_Vegas
│   └───PS-RGB
|          └───SN3_roads_train_AOI_2_Vegas_PS-RGB_img1.tif
           └───SN3_roads_train_AOI_2_Vegas_PS-RGB_img2.tif
           └───---------
|   └———geojson_roads
           └───SN3_roads_train_AOI_2_Vegas_geojson_roads_img1.geojson
           └───SN3_roads_train_AOI_2_Vegas_geojson_roads_img2.geojson
└───AOI_3_Paris
└───AOI_4_Shanghai
└───AOI_4_Shanghai
```
after preprocessing
```
spacenet
|
└───AOI_2_Vegas
│   └───train_RGB
|          └───SN3_roads_train_AOI_2_Vegas_PS-RGB_img1_sat.tif
           └───SN3_roads_train_AOI_2_Vegas_PS-RGB_img2_sat.tif
           └───---------
|   └———label_width24
           └───SN3_roads_train_AOI_2_Vegas_roads_label_img1_mask.tif
           └───SN3_roads_train_AOI_2_Vegas_roads_label_img2_mask.tif
└───AOI_3_Paris
└───AOI_4_Shanghai
└───AOI_4_Shanghai
```

*Download DeepGlobe Road dataset in the following tree structure.*
```
DeepGlobe
   └───104_sat.jpg
   └───104_mask.jpg

```
*Download Massachusetts Roads dataset in the following tree structure.*
```
MassachusettsRoads
   └───train
         └───10078660_15.tiff
         └───10078675_15.tiff
   └───train_labels
         └───10078660_15.tif
         └───10078675_15.tif

```

## Training
Before starting the training, the above data needs to be prepared.
### Modify the configuration file.
[general_cfg.py](cfg_file%2Fgeneral_cfg.py)
Open the configuration file, and then modify the following content.
```
################# you need to change following parameters ##############
DATA_TYPE_flage = '1' #todo you can set 0 1 2 to select SP DP MA dataset
batch_size = 1 #todo 
NUMBER_WORKERS = 0 #todo
image_size = [3,512,512] #todo
SHUFFLE = True
MODEL_SELECT_flag = "17" #todo

LOSS_FUNCTION_flage = "0" #todo you can choose different loss function

LR_SCHEDULER_flage = "6" #todo you can choose different methods for updating learning rate

OPTIMIZER_flage = "6" #todo you can select different OPTIMIZER

train_model_savePath = '/home/wyc/mount/DISCK_1/train_data_cache' #todo choose your save path

#######################  Numeric correspondence  ###########################################################
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
```
### Run the configuration file.
```
cd MSMDFF-NET-main
python cfg_file/general_cfg.py
```
### run traing file
```
cd MSMDFF-NET-main
python train.py
```
# 维护记录
2024.04.13 更新代码运行需要的详细的环境配置信息
