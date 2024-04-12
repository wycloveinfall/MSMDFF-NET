# --coding:utf-8--
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import trange
import os,random
import numpy as np
# import torchvision
"""
author: zch
date: 2022.8.03
"""


class Mytensorboard(object):
    '''
    author:zch
    date:2022.08.04
    '''
    def __init__(self, log_dir):
        _log_dir_root1 = log_dir + '_metric_board/'
        self._metric_board = SummaryWriter(log_dir=_log_dir_root1, flush_secs=60)

    def close_all(self):
        self._metric_board.flush()
        self._metric_board.close()

    def Metrics_graph(self, metrics, metrics_values, epoch: 'int'):
        # metrics can be list to denote several metrics or A single metric parameter
        if isinstance(metrics, list) and isinstance(metrics_values, list):
            if len(metrics) == len(metrics_values):
                for i, metric in enumerate(metrics):
                    try:
                        val = float(metrics_values[i])
                    except TypeError:
                        raise Exception('metrics_values input type error')
                    if 'iou' in metric or 'loss' in metric:
                        self._metric_board.add_scalars('loss_iou/train_test', {metric: val}, epoch)
                    # elif 'test' in metric:
                    #     self._metric_board.add_scalars('Metrics/test', {metric: val}, epoch)
                    else:
                        self._metric_board.add_scalar('Metrics/%s' % metric, val, epoch)
        elif isinstance(metrics, str) and isinstance(metrics_values, float):
            try:
                val = float(metrics_values)
            except TypeError:
                raise Exception('metrics_values input type error')
            self._metric_board.add_scalar('Metrics/%s' % metrics, val, epoch)
        else:
            pass

    def Metrics_all(self, metrics, metrics_values, epoch,write_type='train'):
        # metrics can be list to denote several metrics or A single metric parameter

        _writer = self.writer_root if write_type == 'train' else self.test_metric_board
        for index,metrics_name in enumerate(metrics):
            if 'iou' in metrics_name or 'loss' in metrics_name:
                _writer.add_scalars('a_loss_iou/',{metrics_name: metrics_values[index]}, epoch)
            elif 'NAN' in metrics_name:
                _writer.add_scalars('b_test/', {metrics_name: metrics_values[index]}, epoch)
            else:
                _writer.add_scalars('c_Metric/', {metrics_name: metrics_values[index]}, epoch)

