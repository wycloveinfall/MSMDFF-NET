##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Created by: Wang Yuchuan
## School of Automation Engineering, University of Electronic Science and Technology of China
## Email: wang-yu-chuan@foxmail.com
## Copyright (c) 2017
## for paper: A Multiscale and Multidirection Feature Fusion Network for Road Detection From Satellite Imagery
# https://ieeexplore.ieee.org/document/10477437)
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
import sys
args = sys.argv[1:]
import os
if args == []:
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2'
else:
    os.environ['CUDA_VISIBLE_DEVICES'] = args[0]
# train frame
from utils.frame_work_general import MyFrame
# loss
from utils.loss_fct.dice_loss import DiceLoss
from utils.loss_fct.dice_bce_loss import dice_bce_loss
from utils.loss_fct.MSE_loss import MSE_loss
from utils.loss_fct.Lce_Ldice import Lce_Ldice_loss
# metrics
from utils.metrics_methods.metrics import Evaluator
# tensorboard
from utils.tensorboard_display.tensorboard_base import Mytensorboard
# model_seclect
from model_seclect import Model_get
# model_file
from utils.train_support_fct.model_file_process import loadModelNames

# others
from utils.train_support_fct.others_process import *
import os,json,datetime,time,torch
from tqdm import trange
import numpy as np

cfg_path = './cfg_file/cfg_link_to_data_api.json'

# load config
class CFG(object):
    def __init__(self):
        with open(cfg_path, mode='r', encoding='utf-8') as f:
            cfg_list = json.load(f)
        self.DIR_SETTING = cfg_list[0]['DIR_SETTING']
        self.DATA_SETTING = cfg_list[1]['DATA_SETTING']
        self.OPTIMIZER_SETTING = cfg_list[2]['OPTIMIZER_SETTING']
        self.TRAIN_SETTING = cfg_list[3]['TRAIN_SETTING']
        self.TEST_SETTING = cfg_list[4]['TEST_SETTING']
        self.BEI_ZHU = cfg_list[5]['BEI_ZHU']
        self.CONFIG_NAME = cfg_list[6]['CONFIG_NAME']

cfg = CFG()

if cfg.TRAIN_SETTING['DATA_TYPE'] == 'spacenet_512x512':
    from utils.dataloader._slice_loader.spaceNet_loader import loader_get

elif cfg.TRAIN_SETTING['DATA_TYPE'] == 'DeepGlobe_512x512':
    from utils.dataloader._slice_loader.deepGlobe_loader import loader_get

elif cfg.TRAIN_SETTING['DATA_TYPE'] == 'Massachusetts_512x512':
    from utils.dataloader._slice_loader.Massachusetts_loader import loader_get
else:
    print('dataLoader load error!')
    exit(0)

import random
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True


import argparse
if __name__ == '__main__':
    # 设置随机数种子
    setup_seed(20)
    parser = argparse.ArgumentParser(description='training.')
    parser.add_argument('--local_rank', default=os.getenv('LOCAL_RANK', 0), type=int)
    parser.add_argument('-dis','--DISTRIBUTE', action='store_true', default=False)
    # parser.add_argument('--local_rank', default=os.getenv('LOCAL_RANK', -1), type=int)
    args = parser.parse_args()
    cfg.TRAIN_SETTING['local_rank'] = args.local_rank
    cfg.TRAIN_SETTING["DISTRIBUTE"] = args.DISTRIBUTE

    ####################   init para   ###############################
    beizhu = cfg.BEI_ZHU
    Model_select = cfg.TRAIN_SETTING['MODEL_TYPE']
    TRAIN_IMG_SIZE = cfg.TRAIN_SETTING['TRAIN_IMG_SIZE']
    class_num = cfg.TRAIN_SETTING['CLASS_NUM']

    import platform
    _sysstr = platform.system()
    if (_sysstr == "Windows"):
        print("Call Windows tasks")
    elif (_sysstr == "Linux"):
        print("Call Linux tasks")

    try:
        test_run_frequence = cfg.TEST_SETTING["DISPLAY_FREQ"]
    except:
        test_run_frequence = 1 #兼容之前的版本

    train_size = cfg.DATA_SETTING["train_size"]  # 0.9 和 0.1
    random_state = cfg.DATA_SETTING["random_state"]
    BATCHSIZE_ALL_CARD = cfg.TRAIN_SETTING["BATCH_SIZE"]  # BATCHSIZE_PER_CARD为所有显卡一共的
    TEST_BATCH_SIZE = cfg.TEST_SETTING["BATCH_SIZE"]

    trainDataEnchance = cfg.DATA_SETTING['trainDataEnchance']
    valDataEnchance = cfg.DATA_SETTING['valDataEnchance']
    input_normalization = cfg.DATA_SETTING['input_normalization']
    image_dir = cfg.DIR_SETTING['root_dir']
    # label_dir = cfg.DIR_SETTING['label_dir']

    train_num_workers = cfg.TRAIN_SETTING["NUMBER_WORKERS"]
    val_num_workers = cfg.TEST_SETTING["NUMBER_WORKERS"]
    train_pin_memor = cfg.TRAIN_SETTING["TRAIN_PIN_MEMOR"]
    val_pin_memor = cfg.TEST_SETTING["TEST_PIN_MEMOR"]

    runType = cfg.TRAIN_SETTING["runType"]  #
    # dataReadType = cfg.DATA_SETTING["dataReadType"]
    trainProcessdata_path = cfg.DIR_SETTING['trainProcessdata']
    OPTIMIZER_SETTING = cfg.OPTIMIZER_SETTING

    NAME = '%s-网络配置' % (cfg.CONFIG_NAME)  #
    model_NAME = NAME.split('-')[0]

    ############################################# model slect ##########################################################
    model = Model_get(cfg_path = cfg_path)
    ############################################# loss slect ##########################################################
    loss_type = cfg.OPTIMIZER_SETTING['LOSS_FCT']
    for _loss_fct in [dice_bce_loss,\
                       DiceLoss,\
                       MSE_loss,\
                       Lce_Ldice_loss]:
        if loss_type == _loss_fct.__name__:
            loss_fct = _loss_fct
            break
        else:loss_fct = None
    del _loss_fct


    ############################################# init dir ##########################################################
    modelDir = trainProcessdata_path + '/' + NAME + '/weights/net_sate/'
    best_iou_modelDir = trainProcessdata_path + '/' + NAME + '/weights/best_iou/'
    test_out_src = trainProcessdata_path + '/' + NAME + '/weights/test_out/image_src/'
    test_out_lab = trainProcessdata_path + '/' + NAME + '/weights/test_out/predict/'

    intermediate_modelDir = trainProcessdata_path.replace("\\", "/") + '/' + NAME + '/logs/inter_parameters/'
    outer_parameters = trainProcessdata_path.replace("\\", "/") + '/' + NAME + '/logs/outer_parameters/'
    samples_dictDir = trainProcessdata_path.replace("\\", "/") + '/' + NAME + '/logs/samples_parameters/'

    os.makedirs(modelDir,exist_ok=True)
    os.makedirs(best_iou_modelDir,exist_ok=True)
    os.makedirs(test_out_src,exist_ok=True)
    os.makedirs(test_out_lab,exist_ok=True)

    os.makedirs(intermediate_modelDir,exist_ok=True)
    os.makedirs(outer_parameters,exist_ok=True)
    os.makedirs(samples_dictDir,exist_ok=True)

    ####################   init tensorboard   #######################
    tensorboard_Dir = trainProcessdata_path.replace("\\", "/") + '/' + NAME + '/tensorboard_info/'
    os.makedirs(tensorboard_Dir,exist_ok=True)
    my_board = Mytensorboard(tensorboard_Dir)
    ####################   end  ####################################

    ############################## load log  ################################################################
    log_write_flage = True
    if len(os.listdir(outer_parameters)) == 0:
        log_write_flage = True
        logDir = outer_parameters + NAME + '-' + str(datetime.datetime.now().strftime('%Y-%m-%d %H.%M.%S')) + '.log'
        mylog = open(logDir, 'w',encoding='utf-8')
        best_val_iou = 0
    else:
        log_write_flage = False
        _log_name = [_log_name for _log_name in os.listdir(outer_parameters) if '.log' in _log_name][0]
        logDir = outer_parameters + _log_name
        mylog = open(logDir, 'a+',encoding='utf-8')
    #### end ############################################################################################

    context = 'setting file: %s' % cfg.CONFIG_NAME + '\n' + \
              'Time        : %s' % str(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')) + '\n' + \
              'MODEL_TYPE  : %s' % cfg.TRAIN_SETTING['MODEL_TYPE'] + '\n' + \
              'Data_use    : %s' % image_dir + '\n' + \
              'Data_size    : %s' % cfg.TRAIN_SETTING['TRAIN_IMG_SIZE'] + '\n' + \
              'batch_size    : %s' % BATCHSIZE_ALL_CARD + '\n' + \
              'DataEnchance: %s - %s' % (trainDataEnchance, valDataEnchance) + '\n' + \
              'Input_normal: %s' % input_normalization + '\n' + \
              'OPTIMIZER_SETTING  : %s' % OPTIMIZER_SETTING + '\n' + \
              'Train_save  : %s' % modelDir + '\n' + \
              'other log   : %s' % beizhu + '\n' + \
              '########################### 分割线 #######################\n'

    print(context)
    mylog.write(context) if log_write_flage == True else ''
    mylog.flush()
    # Define Dataloader
    opt = {
    "train_batchsize":BATCHSIZE_ALL_CARD,
    "val_batchsize":TEST_BATCH_SIZE,

    "train_num_workers":cfg.TRAIN_SETTING["NUMBER_WORKERS"],
    "val_num_workers":cfg.TEST_SETTING["NUMBER_WORKERS"],

    "train_pin_memor":cfg.TRAIN_SETTING["TRAIN_PIN_MEMOR"],
    "val_pin_memor":cfg.TEST_SETTING["TEST_PIN_MEMOR"],

    'shuffle':cfg.TRAIN_SETTING["shuffle"]
    }
    train_loader, val_loader = loader_get(cfg.DIR_SETTING['train_sample'],opt)
    cfg.TRAIN_SETTING["train_loader_size"] = len(train_loader)
    cfg.TRAIN_SETTING["val_loader_size"] = len(val_loader)
    ##################################   init model para  ####################################
    image_write_flage = False
    no_optim = 0  # 模型停止 计数器
    total_epoch = 1000
    # train_epoch_best_loss = 10/10
    test_loss = 0
    global_step = float(0)
    save_flag = False
    solver = MyFrame(model, loss_fct, class_num=class_num,cfg=cfg)
    best_train_iou = 0

    best_iou = [[],[]]
    #### load log and model ###############################################################
    model_name, max_num, epoch, lr_current = loadModelNames(modelDir)
    if os.path.exists(model_name):
        solver.load(model_name)
        global_step, epoch, lr_current = max_num, epoch, lr_current
        solver.update_lr(lr_current)
        save_flag = True
        broken_flag = 1
        print('加载 epoch {%d} 成功！\n' % epoch)
        cfg.OPTIMIZER_SETTING["LR_INIT"] = lr_current
    else:
        epoch = 0
        broken_flag = 0
        best_val_iou = 0
        print('无保存模型，将从头开始训练！\n')

    ################ big loop ###############################################################
    evaluator_train = Evaluator(2)
    evaluator_test = Evaluator(2)
    evaluator_train.reset()
    evaluator_test.reset()
    id_dict = {}

    for epoch_ in range(1, total_epoch + 1):
        epoch = epoch + 1
        print('\n\n---------- Epoch:' + str(epoch) + ' ----------')
        train_epoch_loss = []
        train_epoch_iou = []
        train_epoch_id = []
        test_epoch_loss = []
        test_epoch_iou = []
        test_epoch_id = []
        predict_out = test_out_lab + 'epoch_%d' % epoch
        mkdir(predict_out)


        data_loader_iter = iter(train_loader)

        solver.net.train()  # 使能 BN层 drouptout层的 随机跳动
        with trange(len(train_loader)) as t:  # 实现进度条
            epoch_time = np.zeros([len(train_loader)])
            for index in t:
                time_start = time.time()
                # train_sample = data_loader_iter.next()
                train_sample = next(data_loader_iter)
                if not list(train_sample['image'].shape)[-1] == cfg.TRAIN_SETTING['TRAIN_IMG_SIZE'][-1]: print('data_size error!!')
                solver.set_input(train_sample)
                batch_loss_n, pred = solver.optimize(index+1,epoch)

                with torch.no_grad():
                    # metric
                    batch_loss_n = batch_loss_n.cpu().data.numpy()
                    pred = pred.cpu().data.numpy().squeeze()
                    target = train_sample['label'].cpu().data.numpy().squeeze().astype(np.uint8)
                    pred[pred < 0.5] = 0
                    pred[pred >= 0.5] = 1
                    batch_iou_n = evaluator_train.add_batch_and_return_iou(target, pred)

                    for train_single_id, train_single_loss, train_single_iou \
                            in zip(train_sample['id'], batch_loss_n, batch_iou_n):
                        train_epoch_id.append(train_single_id)
                        try:
                           train_epoch_loss.append(train_single_loss[0])
                        except:
                           train_epoch_loss.append(train_single_loss)
                        train_epoch_iou.append(train_single_iou)
                    train_batch_loss_show = batch_loss_n.mean(0)
                    train_batch_iou_show = batch_iou_n.mean(0)
                    train_epoch_loss_show = np.mean(train_epoch_loss)
                    train_epoch_iou_show = np.mean(train_epoch_iou)
                    global_step = global_step + 1

                    time_end = time.time()
                    epoch_time[index] = time_end - time_start
                    last_time = ((len(train_loader)-index-1) + (200-epoch-1)*len(train_loader)) *np.mean(epoch_time[:index+1])
                    m,s = divmod(last_time,60)
                    h,m = divmod(m,60)
                    spend_time = '%dh:%dm:%ds'%(h,m,s)

                    t.set_description('training...loss:%.8f - iou:%.8f' % (train_batch_loss_show, train_batch_iou_show))
                    t.set_postfix(step=global_step, loss=train_epoch_loss_show, iou=train_epoch_iou_show, spend_time=spend_time)

        # Fast test during the training
        Acc = evaluator_train.Pixel_Accuracy()
        Acc_class = evaluator_train.Pixel_Accuracy_Class()
        mIoU = evaluator_train.Mean_Intersection_over_Union()
        IoU = evaluator_train.Intersection_over_Union()
        Precision = evaluator_train.Pixel_Precision()
        Recall = evaluator_train.Pixel_Recall()
        F1 = evaluator_train.Pixel_F1()
        evaluator_train.reset()
        print('train:')
        print("Acc:{}, Acc_class:{}, mIoU:{}, IoU:{}, Precision:{}, Recall:{}, F1:{}"
              .format(Acc, Acc_class, mIoU, IoU, Precision, Recall, F1))

        val_data_loader_iter = iter(val_loader)
        solver.net.eval()# 关闭 BN层 drouptout层的 随机跳动
        with trange(len(val_loader)) as t:
            for index in t:
                # val_sample = val_data_loader_iter.next()
                val_sample = next(val_data_loader_iter)
                if val_sample['label'].shape[1] > 3:
                    val_sample['label'] = val_sample['label'][:, :3, :, :]  # 有的图像是四通道
                predict_n, batch_loss_n = solver.test_one_img(val_sample) #

                # metric
                pred = predict_n.cpu().data.numpy().squeeze()
                target = val_sample['label'].cpu().data.numpy().squeeze().astype(np.uint8)
                pred[pred < 0.5] = 0
                pred[pred >= 0.5] = 1
                batch_iou_n = evaluator_test.add_batch_and_return_iou(target, pred) #IoU = evaluator.Intersection_over_Union()
                # IOU LOSS 处理 ########################################################################################
                for test_single_id, test_single_loss, test_single_iou in zip(val_sample['id'], batch_loss_n, batch_iou_n):
                    test_epoch_id.append(test_single_id)
                    try:
                       test_epoch_loss.append(test_single_loss[0])
                    except:
                        test_epoch_loss.append(test_single_loss)
                    test_epoch_iou.append(test_single_iou)

                test_batch_loss_show = batch_loss_n.mean(0)
                test_batch_iou_show = batch_iou_n.mean(0)
                test_epoch_loss_show = np.mean(test_epoch_loss)
                test_epoch_iou_show = np.mean(test_epoch_iou)

                t.set_description('testing...loss:%.8f - iou:%.8f' % (test_batch_loss_show, test_batch_iou_show))
                t.set_postfix(val_step=index, test_loss=test_epoch_loss_show, test_iou=test_epoch_iou_show,
                              lr=solver.lr_current)
            save_flag = False


        # Fast test during the testing
        Acc_t = evaluator_test.Pixel_Accuracy()
        Acc_class_t = evaluator_test.Pixel_Accuracy_Class()
        mIoU_t = evaluator_test.Mean_Intersection_over_Union()
        IoU_t = evaluator_test.Intersection_over_Union()
        Precision_t = evaluator_test.Pixel_Precision()
        Recall_t = evaluator_test.Pixel_Recall()
        F1_t = evaluator_test.Pixel_F1()
        evaluator_test.reset()

        if best_val_iou < IoU_t:#todo 该参数未保存
            best_val_iou = IoU_t
            best_iou[1].append([epoch, best_val_iou])
        if best_train_iou < IoU:
            best_train_iou = IoU
            best_iou[0].append([epoch, best_train_iou])

        ################################## 模型权重级训练日志保存 与 学习率设置    #########################################
        now_time = str(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        print('(%d)--epoch:' % broken_flag, epoch, '  --time:', now_time,
              '  --train_loss:', train_epoch_loss_show, ' --val_loss:', test_epoch_loss_show, \
              ' --train_iou:', train_epoch_iou_show, ' --val_iou:', test_epoch_iou_show,
              '  --lr:%s' % str(solver.lr_current))

        print('Validation:')
        print("Acc:{}, Acc_class:{}, mIoU:{}, IoU:{}, Precision:{}, Recall:{}, F1:{}"
              .format(Acc_t, Acc_class_t, mIoU_t, IoU_t, Precision_t, Recall_t, F1_t))

        #写入日志
        mylog.write('(%d)--epoch:'%broken_flag + str(epoch).rjust(3,'0') + '  --time:' + now_time + \
                    '  --train_loss:' + str(train_epoch_loss_show).ljust(10,'0') + '  --val_loss:' + str(test_epoch_loss_show).ljust(10,'0') + \
                    '  --train_iou:%.8f'%(train_epoch_iou_show) + '  --val_iou:%.8f'%(test_epoch_iou_show) + \
                    '  --lr:%.16f' %(solver.lr_current) + \
                    '  --TestAcc:{}  --Acc_class:{}  --mIoU:{}  --IoU:{}  --Precision:{}  --Recall:{}  --F1:{}' \
                        .format(Acc_t, Acc_class_t, mIoU_t, IoU_t, Precision_t, Recall_t, F1_t,16) + \
                    # '  --TrainAcc:{}  --Acc_class:{}  --mIoU:{}  --IoU:{}  --Precision:{}  --Recall:{}  --F1:{}' \
                    #     .format(Acc, Acc_class, mIoU, IoU, Precision, Recall, F1) + \
                    '\n')
        # 清除 断点标记
        broken_flag = 0
        # 刷新缓冲区
        mylog.flush()
        # 保存模型
        solver.save(modelDir + str(global_step) + '_$%d$' % epoch + NAME + '$%.8f$.th' % solver.lr_current)


        my_board.Metrics_graph(metrics=['train_iou', 'train_loss', 'test_iou', 'test_loss', 'Acc',
                                        'Acc_class', 'mIoU', 'IoU', 'Precision', 'Recall', 'F1'],
                           metrics_values=[train_epoch_iou_show, train_epoch_loss_show,
                                           test_epoch_iou_show, test_epoch_loss_show,
                                           Acc_t, Acc_class_t, mIoU_t, IoU_t, Precision_t, Recall_t, F1_t],
                                           epoch=epoch)
        ############################################################################################################

        ############################################################################################################


    print(mylog, 'Finish!')
    print('Finish!')
    mylog.close()

