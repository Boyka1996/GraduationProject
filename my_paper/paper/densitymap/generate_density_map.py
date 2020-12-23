#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time       : 2020/12/23 22:01
@Author     : Boyka
@Contact    : upcvagen@163.com
@File       : generate_density_map.py
@Project    : GraduationProject
@Description:
"""
import h5py
import scipy.io as io
import PIL.Image as Image
import numpy as np
import os
import glob
from matplotlib import pyplot as plt
from scipy.ndimage.filters import gaussian_filter
import scipy
import scipy.spatial as spatial
import json
from matplotlib import cm as CM
from image import *
# from model import CSRNet
import torch


def gaussian_filter_density(gt):
    print(gt.shape)
    density = np.zeros(gt.shape, dtype=np.float32)
    gt_count = np.count_nonzero(gt)
    if gt_count == 0:
        return density
    # print(np.nonzero(gt))
    print(len(np.nonzero(gt)[0]))
    print(list(zip(np.nonzero(gt)[1], np.nonzero(gt)[0])))
    pts = np.array(list(zip(np.nonzero(gt)[1], np.nonzero(gt)[0])))
    leafsize = 2048
    # build kdtree
    tree = scipy.spatial.KDTree(pts.copy(), leafsize=leafsize)
    # query kdtree
    distances, locations = tree.query(pts, k=4)

    print('generate density...')
    for i, pt in enumerate(pts):
        pt2d = np.zeros(gt.shape, dtype=np.float32)
        pt2d[pt[1], pt[0]] = 1.
        if gt_count > 1:
            sigma = (distances[i][1] + distances[i][2] + distances[i][3]) * 0.1
        else:
            sigma = np.average(np.array(gt.shape)) / 2. / 2.  # case: 1 point
        density += scipy.ndimage.filters.gaussian_filter(pt2d, sigma, mode='constant')
    print('done.')
    return density
mat = io.loadmat('E:/Dataset/密集人群计数/ShanghaiTech/part_A_final/train_data/ground_truth/GT_IMG_43.mat')
img= plt.imread('1.jpg')
k = np.zeros((img.shape[0],img.shape[1]))
gt = mat["image_info"][0,0][0,0][0]
for i in range(0,len(gt)):
    if int(gt[i][1])<img.shape[0] and int(gt[i][0])<img.shape[1]:
        k[int(gt[i][1]),int(gt[i][0])]=1
k = gaussian_filter_density(k)
with h5py.File('1.h5', 'w') as hf:
        hf['density'] = k
gt_file = h5py.File('1.h5','r')
groundtruth = np.asarray(gt_file['density'])
plt.imshow(groundtruth,cmap=CM.jet)
plt.show()
plt.savefig('heatmap.png')