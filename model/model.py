#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2020/6/1 下午8:11
# @Author  : Boyka
# @Email   : upcvagen@163.com
# @File    : model.py
# @Software: PyCharm
import torch.nn as nn


class MyModel(nn.Module):
    def __init__(self):
        self.brunch1 = nn.Sequential(nn.Conv2d(in_channels=3, out_channels=19, kernel_size=6, stride=1),
                                     nn.MaxPool2d(2),
                                     nn.Conv2d)
