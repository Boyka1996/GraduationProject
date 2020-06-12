#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2020/6/10 上午10:34
# @Author  : Boyka
# @Email   : upcvagen@163.com
# @File    : mcnn.py
# @Software: PyCharm

import torch.nn as nn
import torch
from mcnn.mcnn_model import MCNNModel
# from mcnn.mcnn_model import MCNNModel


class MCNN(nn):
    def __init__(self):
        super(MCNN, self).__init__()
        self.model = MCNNModel()
        self.mse_loss = nn.MSELoss()

    def loss(self):
        return self.mse_loss

    def forward(self, in_data, gt_data):
        pre_density_map = self.model(in_data)
        if self.training:
            self.mse_loss = self.get_loss(pre_density_map, gt_data)
        return pre_density_map

