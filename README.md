# MSMDFF-NET
the code for paper "A Multi-Scale and Multi-Direction Feature Fusion Network for Road Detection From Satellite Imagery".  The code will be made public after the paper is accepted.
# sorry
I'm very sorry, but I've fallen ill. The release of the code requires organizing references to previous code and explaining the power relations, so the release time will be delayed. The exact time will be before June. Anyone can contact me to get an unpublished version of the code.

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
- Python 3.8.16
- [PyTorch](https://pytorch.org/get-started/previous-versions/) 1.8.1

### Data Preparation
#### PreProcess SpaceNet Dataset
- Convert SpaceNet 11-bit images to 8-bit Images. [1.convert_to_8bit.py](data%2Fdata_tools%2F1.convert_to_8bit.py)
- Create road centerline.[2.cerete_label.py](data%2Fdata_tools%2F2.cerete_label.py)
- Create road masks.[3.fill_center_line.py](data%2Fdata_tools%2F3.fill_center_line.py)
- select Train files. [4.choose_train_flie.py](data%2Fdata_tools%2F4.choose_train_flie.py)

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
|          └───SN3_roads_train_AOI_2_Vegas_PS-RGB_img1.tif
           └───SN3_roads_train_AOI_2_Vegas_PS-RGB_img2.tif
           └───---------
|   └———label_width24
           └───SN3_roads_train_AOI_2_Vegas_roads_label_img1.tif
           └───SN3_roads_train_AOI_2_Vegas_roads_label_img2.tif
└───AOI_3_Paris
└───AOI_4_Shanghai
└───AOI_4_Shanghai
```

*Download DeepGlobe Road dataset in the following tree structure.*
```
deepglobe
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
## Code organization is still in progress.

-2024.04.10 networks
