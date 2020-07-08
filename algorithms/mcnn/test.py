#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2020/7/4 下午2:23
# @Author  : Boyka
# @Email   : upcvagen@163.com
# @Software: PyCharm

import numpy as np
import torch

density_map = np.load('/home/chase/datasets/crowd_counting/ShanghaiTech/part_B_final/train_data/npy/IMG_5.npy')
# utils.show_desity_map(density_map)
a = np.zeros(shape=(3, 4, 5))
a = np.swapaxes(a, 0, 2)
print(a.shape)
device='cuda' if torch.cuda.is_available() else 'cpu'
print(torch.cuda.get_device_capability(device), torch.cuda.get_device_name(device))
