#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2020/5/22 下午2:19
# @Author  : Boyka
# @Email   : upcvagen@163.com
# @File    : read_from_npy.py
# @Software: PyCharm

import numpy as np

npy_path = '/home/chase/datasets/crowd_counting/ShanghaiTech/part_A_final/test_data/1/val_gt/GT_IMG_1_3.npy'
matrix = np.load(npy_path)
print(matrix.shape)
