#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2020/5/22 下午2:21
# @Author  : Boyka
# @Email   : upcvagen@163.com
# @File    : test2_backup.py
# @Software: PyCharm

import argparse
import json
import logging
import os
import time
import cv2

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy.ndimage import filters
from sklearn.neighbors import NearestNeighbors

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


def parse_args():
    parser = argparse.ArgumentParser(description="Path configuration")
    parser.add_argument(
        '--image_path',
        dest='image_path',
        help='',
        default='/home/chase/datasets/crowd_counting/mall/frames',
        # default=None,
        type=str
    )

    parser.add_argument(
        '--json_path',
        dest='json_path',
        help='',
        default='/home/chase/datasets/crowd_counting/mall/json',
        # default=None,
        type=str
    )
    parser.add_argument(
        '--npy_path',
        dest='npy_path',
        help='',
        default='/home/chase/datasets/crowd_counting/mall/npy_kernel_size_5_sigma',
        # default=None,
        type=str
    ),
    parser.add_argument(
        '--data_set_info',
        dest='data_set_info',
        help='',
        default='/home/chase/datasets/crowd_counting/mall/mall_info.json',
        # default=None,
        type=str
    )
    return parser.parse_args()


def create_density(gts, d_map_h, d_map_w):
    start_time = time.time()
    res = np.zeros(shape=[d_map_h, d_map_w])
    bool_res = (gts[:, 0] < d_map_w) & (gts[:, 1] < d_map_h)
    for k in range(len(gts)):
        gt = gts[k]
        if bool_res[k]:
            res[int(gt[1])][int(gt[0])] = 1
    pts = np.array(list(zip(np.nonzero(res)[1], np.nonzero(res)[0])))
    neighbors = NearestNeighbors(n_neighbors=4, algorithm='kd_tree', leaf_size=1200)
    neighbors.fit(pts.copy())
    distances, _ = neighbors.kneighbors()
    map_shape = [d_map_h, d_map_w]
    density = np.zeros(shape=map_shape, dtype=np.float32)
    sigmas = distances.sum(axis=1) * 0.075
    for i in range(len(pts)):
        pt = pts[i]
        pt2d = np.zeros(shape=map_shape, dtype=np.float32)
        pt2d[pt[1]][pt[0]] = 1
        radius = 5 * sigmas[i]
        xmin = int(max(0, pt[0] - radius))
        xmax = int(min(pt[0] + radius, d_map_w))
        ymin = int(max(0, pt[1] - radius))
        ymax = int(min(pt[1] + radius, d_map_h))
        # print(xmin)
        # print(xmax)q
        # print(ymin)
        # print(ymax)
        # print(pt2d[xmin:xmax, ymin:ymax].shape)
        # one_map = cv2.GaussianBlur(pt2d, ksize=(9, 9), sigmaX=0, sigmaY=0)
        one_map = filters.gaussian_filter(pt2d[xmin:xmax, ymin:ymax], sigmas[i], mode='constant')
        density[xmin:xmax, ymin:ymax] += one_map
    logger.info("***********************************")
    logger.info(time.time() - start_time)
    logger.info(np.sum(density))
    logger.info("***********************************")
    plt.imshow(density)
    plt.show()
    return density


def get_file_list(file_path):
    return zip(os.listdir(file_path), [os.path.join(file_path, file) for file in os.listdir(file_path)])


if __name__ == '__main__':
    data_args = parse_args()
    for image_name, image_path in get_file_list(data_args.image_path):
        logger.info(image_name)
        pil_img = Image.open(image_path)
        shape = pil_img.size
        print(shape)
        json_path = os.path.join(data_args.json_path, image_name.replace('.jpg', '.json'))
        with open(json_path, 'r') as fr:
            points = json.load(open(json_path)).get('points')
        logger.info(len(points))
        points = np.array(points)
        den_map = create_density(points, pil_img.size[0], pil_img.size[1])
        # d_map_h = math.floor(math.floor(float(pil_img.size[0]) / 2.0) / 2.0)
        # d_map_w = math.floor(math.floor(float(pil_img.size[1]) / 2.0) / 2.0)
        # den_map = create_density(points / 4, d_map_h, d_map_w)
        #
        # p_h = math.floor(float(pil_img.size[0]) / 3.0)
        # p_w = math.floor(float(pil_img.size[1]) / 3.0)
        # d_map_ph = math.floor(math.floor(p_h / 2.0) / 2.0)
        # d_map_pw = math.floor(math.floor(p_w / 2.0) / 2.0)
