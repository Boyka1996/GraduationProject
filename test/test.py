#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2020/6/1 下午4:37
# @Author  : Boyka
# @Email   : upcvagen@163.com
# @File    : test.py
# @Software: PyCharm
import numpy as np
file_path='/home/chase/datasets/crowd_counting/NWPU/npy/0032.npy'
density_map=np.load(file_path)
print(density_map.shape)