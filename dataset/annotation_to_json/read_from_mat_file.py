#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2020/5/28 下午3:14
# @Author  : Boyka
# @Email   : upcvagen@163.com
# @File    : read_from_mat_file.py
# @Software: PyCharm

import argparse
import json
import logging
import os

import scipy.io as sci_io
def shanghai_crowd_points(mat_file_path):
    """
    :param mat_file_path: Mat file path
    :return: Numpy.ndarray of points
    """
    mat_info = sci_io.loadmat(mat_file_path)
    crowd_points = mat_info.get('image_info')[0][0][0][0][0]
    return crowd_points.tolist()

file_path='/home/chase/datasets/crowd_counting/ShanghaiTech/part_A_final/train_data/ground_truth/GT_IMG_76.mat'
print(shanghai_crowd_points(file_path))