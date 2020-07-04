#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2020/7/4 下午2:23
# @Author  : Boyka
# @Email   : upcvagen@163.com
# @Software: PyCharm

import utils
import numpy as np
density_map=np.load('/home/chase/datasets/crowd_counting/ShanghaiTech/part_B_final/train_data/npy/IMG_5.npy')
utils.show_desity_map(density_map)