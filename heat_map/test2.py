#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/5/2 15:02
# @Author  : Boyka
# @Email   : upcvagen@163.com
# @File    : test2.py

import math
import os

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
from PIL import Image
from scipy.ndimage import filters
from sklearn.neighbors import NearestNeighbors


def create_density(gts, d_map_h, d_map_w):
    # print(gts)
    res = np.zeros(shape=[d_map_h, d_map_w])
    bool_res = (gts[:, 0] < d_map_w) & (gts[:, 1] < d_map_h)
    for k in range(len(gts)):
        gt = gts[k]
        if (bool_res[k] == True):
            res[int(gt[1])][int(gt[0])] = 1
    pts = np.array(list(zip(np.nonzero(res)[1], np.nonzero(res)[0])))
    neighbors = NearestNeighbors(n_neighbors=4, algorithm='kd_tree', leaf_size=1200)
    neighbors.fit(pts.copy())
    distances, points_ = neighbors.kneighbors()

    map_shape = [d_map_h, d_map_w]
    density = np.zeros(shape=map_shape, dtype=np.float32)
    sigmas = distances.sum(axis=1)/2
    for i in range(len(pts)):
        pt = pts[i]
        pt2d = np.zeros(shape=map_shape, dtype=np.float32)
        pt2d[pt[1]][pt[0]] = 1
        # starttime = datetime.datetime.now()
        # fig1 = plt.figure('fig1')
        # plt.imshow(filters.gaussian_filter(pt2d, sigmas[i], mode='constant'))
        # plt.show()
        density += filters.gaussian_filter(pt2d, sigmas[i], mode='constant')
    print(np.sum(density))
    return density


if __name__ == '__main__':
    train_img = '/home/chase/datasets/crowd_counting/ShanghaiTech/part_A_final/test_data/images'
    train_gt = '/home/chase/datasets/crowd_counting/ShanghaiTech/part_A_final/test_data/ground_truth'
    out_path = '/home/chase/datasets/crowd_counting/ShanghaiTech/part_A_final/test_data/1/'
    validation_num = 15

    img_names = os.listdir(train_img)
    num = len(img_names)
    num_list = np.arange(1, num + 1)
    # random.shuffle(num_list)
    global_step = 1
    for i in num_list:
        full_img = train_img + '/IMG_' + str(i) + '.jpg'
        full_gt = train_gt + '/GT_IMG_' + str(i) + '.mat'
        img = mpimg.imread(full_img)
        data = sio.loadmat(full_gt)
        gts = data['image_info'][0][0][0][0][0]  # shape like (num_count, 2)
        print(len(gts))
        count = 1

        # fig1 = plt.figure('fig1')
        # plt.imshow(img)
        shape = img.shape
        if (len(shape) < 3):
            img = img.reshape([shape[0], shape[1], 1])

        d_map_h = math.floor(math.floor(float(img.shape[0]) / 2.0) / 2.0)
        d_map_w = math.floor(math.floor(float(img.shape[1]) / 2.0) / 2.0)
        # starttime = datetime.datetime.now()
        # den_map = gaussian_filter_density(res)
        if (global_step == 4):
            print(1)
        # den_map = create_density(gts, img.shape[0], img.shape[1])
        #
        den_map = create_density(gts / 4, d_map_h, d_map_w)
        print(den_map)
        fig1 = plt.figure('fig1')
        plt.imshow(den_map)
        plt.show()