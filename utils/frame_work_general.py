# -*-coding:utf-8-*-
# !/usr/bin/env python
# @Time    : 2023/3/11 下午6:39
# @Author  : 王玉川 uestc
# @File    : frame_work_general.py
# @Description :
#  python3
# -*- coding: utf-8 -*-
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Created by: Wang Yuchuan
## School of Automation Engineering, University of Electronic Science and Technology of China
## Email: wang-yu-chuan@foxmail.com
## Copyright (c) 2017
## for paper: A Multiscale and Multidirection Feature Fusion Network for Road Detection From Satellite Imagery
# https://ieeexplore.ieee.org/document/10477437)
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
import torch
from torch.autograd import Variable as V
from torch.optim import lr_scheduler

import cv2,os
import numpy as np

from utils.learnlingrate_methods.lr_scheduler import LR_Scheduler
from torch.autograd import Variable
import torch.distributed as dist


class MyFrame():
    def __init__(self, net, loss=None,class_num = 1,cfg=None, mylog='', evalmode = False,is_useGpu = True):
        # assert not cfg==None
        self.cfg = cfg
        self.log = mylog
        self.loss = loss() if not loss == None else ''

        self.lr_current = self.cfg.OPTIMIZER_SETTING["LR_INIT"] if not cfg == None else 0.001

        self.loss_calculate_device = 0
        self.is_useGpu = is_useGpu

        self.best_pred = 0

        try:
            DISTRIBUTE = self.cfg.TRAIN_SETTING["DISTRIBUTE"]
        except:
            DISTRIBUTE = False

        if evalmode:
            net = net().cuda()
            self.net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count())) \
                       if is_useGpu == True else net
            # self.optimizer = torch.optim.Adam(params=self.net.parameters(), lr=self.lr_current)
        else:
            if DISTRIBUTE:
                ####################### 分布式训练加载模型 ####################################
                # dist.init_process_group(backend='nccl')

                rank, world_size = dist.get_rank(), dist.get_world_size()
                # rank = cfg.TRAIN_SETTING['local_rank']
                device_id = rank % torch.cuda.device_count()
                device = torch.device(device_id)
                torch.cuda.set_device(device)
                torch.backends.cudnn.allow_tf32 = False
                torch.backends.cuda.matmul.allow_tf32 = False
                net = net().cuda() #把这句话放在这里很关键
                self.net = torch.nn.parallel.DistributedDataParallel(net, device_ids=[device_id],
                                                          output_device=cfg.TRAIN_SETTING['local_rank'] , find_unused_parameters=True)
            else:
                net = net().cuda()
                self.net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()),
                                                      output_device=self.loss_calculate_device)
            # self.net = torch.compile(self.net) if float(torch.__version__[0:3])>=2 else self.net#2023.06.15add
            # self.net = self.net#2023.06.15add
            if self.cfg.OPTIMIZER_SETTING["OPTIMIZER"] == 'Adam(AMSGrad)':

                self.optimizer = torch.optim.Adam(params=self.net.parameters(), lr=self.lr_current)
            elif self.cfg.OPTIMIZER_SETTING["OPTIMIZER"] == 'SGD':
                self.optimizer = torch.optim.SGD(params=self.net.parameters(), lr=self.lr_current,momentum=0.9, weight_decay=0.0001)

            else:
                print('error self.optimizer')
                exit(0)
            self.scheduler = LR_Scheduler(self.cfg.OPTIMIZER_SETTING["LR_SCHEDULER"],
                                          self.cfg.OPTIMIZER_SETTING["LR_INIT"],
                                          self.cfg.TRAIN_SETTING["EPOCHS"],
                                          self.cfg.TRAIN_SETTING["train_loader_size"],
                                          local_rank = cfg.TRAIN_SETTING['local_rank'])
            print('Using {} LR Scheduler!'.format(self.cfg.OPTIMIZER_SETTING["LR_SCHEDULER"])) if cfg.TRAIN_SETTING['local_rank']==0 else ''


    def to_variable(self,tensor, requires_grad=False):
        return Variable(tensor.long().cuda(), requires_grad=requires_grad)

    def set_input(self, train_sample):
        self.img = train_sample['image']
        self.mask = train_sample['label']
        self.img_id = train_sample['id']
        self.img_name = self.img_id[0].replace('\\', '/').split('/')[-1].split('.')[0]
        # 保存测试
        # from PIL import Image
        # image = self.img.data.numpy().squeeze().transpose([1,2,0])
        # mask = self.mask.data.numpy().squeeze()
        # img1 = Image.fromarray(image.astype(np.uint8))
        # mask = Image.fromarray(mask.astype(np.uint8)*255)
        # img1.save('/home/wychasee/图片/01/%s.png'%self.img_name)
        # mask.save('/home/wychasee/图片/01/%s_mask.png'%self.img_name)

        if self.is_useGpu:
            self.img = self.img.cuda()
            self.mask = self.mask.cuda() #
    def optimize(self,i,epoch):

        self.lr_current = self.scheduler(self.optimizer, i, epoch,debuge=False)
        self.optimizer.zero_grad()                   # 清除优化器 状态
        pred = self.net.forward(self.img)            # 计算预测
        loss_batch,loss = self.loss(pred, self.mask) # 计算loss
        loss.backward()        # 误差反向传播
        self.optimizer.step()  # 更新参数
        return loss_batch,pred




    def test_one_img_for_feature(self, val_sample):  #注释了一部分的
        with torch.no_grad():
            img = val_sample['image']
            mask = val_sample['label']
            img_id = val_sample['id']

            if self.is_useGpu:
                img = img.cuda()
                mask = mask.cuda()  #
            # hook ####
            self.img_name = img_id[0].replace('\\', '/').split('/')[-1].split('.')[0]
            if self.model_name and self.img_name:
                self.pre_Dir = self.save_Dir + '/' + 'model-%s/image-%s' % (self.model_name, self.img_name)
                if not os.path.exists(self.pre_Dir):
                    os.makedirs(self.pre_Dir)
            self.hookAllLayers()
            pred = self.net(img)
            out_im = pred.cpu().data.numpy().squeeze()*255
            cv2.imwrite('%s/input.png'%self.pre_Dir,img[1][0].cpu().data.numpy().astype(np.uint8))
            cv2.imwrite('%s/result.png'%self.pre_Dir,out_im[1].astype(np.uint8))

            # pred = self.net(img)

            loss_batch,loss = self.loss(pred, mask)  # 计算loss
        batch_loss = loss_batch.cpu().data.numpy() if self.is_useGpu else loss_batch.data.numpy()

        # self.lyr_names = []
        return pred, batch_loss



    def test_one_img(self, val_sample):  #注释了一部分的
        with torch.no_grad():
            img = val_sample['image']
            mask = val_sample['label']
            img_id = val_sample['id']
            if self.is_useGpu:
                img = img.cuda()
                mask = mask.cuda()  #

            pred = self.net(img)

            loss_batch,loss = self.loss(pred, mask)  # 计算loss
        batch_loss = loss_batch.cpu().data.numpy() if self.is_useGpu else loss_batch.data.numpy()

        # self.lyr_names = []
        return pred, batch_loss
    def test_one_img_res(self, val_sample):  #注释了一部分的
        with torch.no_grad():
            img = val_sample['image']
            mask = val_sample['label']
            img_id = val_sample['id']
            if self.is_useGpu:
                img = img.cuda()
                mask = mask.cuda()  #

            pred,pred_res = self.net(img)

            loss_batch,loss = self.loss(pred, mask)  # 计算loss
        batch_loss = loss_batch.cpu().data.numpy() if self.is_useGpu else loss_batch.data.numpy()

        # self.lyr_names = []
        return pred+pred_res, batch_loss

    def save(self, path):
        torch.save(self.net.state_dict(), path)

    def load(self, path):
        self.model_name = path.replace('\\', '/').split('/')[-1].split('$')[2] #2023.09.05 fix
        checkpoint_all = torch.load(path)
        self.net.load_state_dict(checkpoint_all)


    def update_lr(self, new_lr, factor=False):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = new_lr
        # print("here is not error  ######1#######")
        if self.cfg.TRAIN_SETTING['local_rank'] == 0:
            self.log.write('Note: update learning rate: %.10f -> %.10f\n' % (self.lr_current, new_lr)) if not self.log == '' else 0
            print('update learning rate: %.10f -> %.10f' % (self.lr_current, new_lr))
            self.lr_current = new_lr


    def lr_strategy(self):
        # scheduler = lr_scheduler.StepLR(self.optimizer, step_size=30, gamma=0.1)
        # scheduler = lr_scheduler.MultiStepLR(self.optimizer, [30, 80], 0.1)
        scheduler = lr_scheduler.ExponentialLR(self.optimizer, gamma=0.9)
        return scheduler



    def test_batch(self):
        self.forward(volatile=True)
        mask =  self.net.forward(self.img).cpu().data.numpy().squeeze(1)
        mask[mask>0.5] = 1
        mask[mask<=0.5] = 0

        return mask, self.img_id

    def test_one_img_from_path(self, path):
        img = cv2.imread(path)
        img = np.array(img, np.float32)/255.0 * 3.2 - 1.6
        img = V(torch.Tensor(img).cuda())

        mask = self.net.forward(img).squeeze().cpu().data.numpy()#.squeeze(1)
        mask[mask>0.5] = 1
        mask[mask<=0.5] = 0
        return mask
