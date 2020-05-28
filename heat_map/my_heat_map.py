#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2020/5/26 上午11:24
# @Author  : Boyka
# @Email   : upcvagen@163.com
# @File    : my_heat_map.py
# @Software: PyCharm

import argparse
import json
import logging
import os
import time

import cv2
import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors import NearestNeighbors

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


#
# def parse_args():
#     parser = argparse.ArgumentParser(description="Path configuration")
#     parser.add_argument(
#         '--image_path',
#         dest='image_path',
#         help='',
#         default='/home/chase/datasets/crowd_counting/mall/frames',
#         # default=None,
#         type=str
#     )
#
#     parser.add_argument(
#         '--json_path',
#         dest='json_path',
#         help='',
#         default='/home/chase/datasets/crowd_counting/mall/json',
#         # default=None,
#         type=str
#     )
#     parser.add_argument(
#         '--npy_path',
#         dest='npy_path',
#         help='',
#         default='/home/chase/datasets/crowd_counting/mall/npy',
#         # default=None,
#         type=str
#     )
#     return parser.parse_args()
#


def parse_args():
    parser = argparse.ArgumentParser(description="Path configuration")
    parser.add_argument(
        '--image_path',
        dest='image_path',
        help='',
        default='/home/chase/datasets/crowd_counting/NWPU/images',
        # default=None,
        type=str
    )

    parser.add_argument(
        '--json_path',
        dest='json_path',
        help='',
        default='/home/chase/datasets/crowd_counting/NWPU/json',
        # default=None,
        type=str
    )
    parser.add_argument(
        '--npy_path',
        dest='npy_path',
        help='',
        default='/home/chase/datasets/crowd_counting/NWPU/npy',
        # default=None,
        type=str
    )
    return parser.parse_args()


def create_density(cv_image, points, density_map_rows, density_map_cols):
    """

    :param cv_image:
    :param points: (x,y) == (col_id,row_id)
    :param density_map_rows: height == rows
    :param density_map_cols: width == cols
    :return:
    """
    density = np.zeros(shape=(density_map_rows, density_map_cols), dtype=np.float32)
    if len(points) < 5:
        return density
    points = points[:, [1, 0]]
    start_time = time.time()
    neighbors = NearestNeighbors(n_neighbors=4, algorithm='kd_tree', leaf_size=1200)
    neighbors.fit(points.copy())
    distances, _ = neighbors.kneighbors()

    sigmas = distances.sum(axis=1) * 0.075
    for i in range(len(points)):
        point = points[i]
        single_heat_map = np.zeros(shape=(density_map_rows, density_map_cols), dtype=np.float32)
        single_heat_map[point[0]][point[1]] = 1
        # sigmas[i] = int(sigmas[i])
        sigma = sigmas[i].astype(np.int16)
        # Scale adaptive Gaussian kernel
        if sigma % 2 == 0:
            sigma -= 1
        # radius = 100 * sigma
        radius = 49 * sigma
        row_min = max(0, point[0] - radius)
        row_max = min(point[0] + radius, density_map_rows)
        col_min = max(0, point[1] - radius)
        col_max = min(point[1] + radius, density_map_cols)
        single_heat_map = cv2.GaussianBlur(single_heat_map[row_min:row_max, col_min:col_max],
                                           ksize=(3 * sigma, 3 * sigma), sigmaX=0, sigmaY=0)
        density[row_min:row_max, col_min:col_max] += single_heat_map
    logger.info("***********************************")
    logger.info(time.time() - start_time)
    logger.info(len(points))
    logger.info(np.sum(density))
    logger.info("***********************************")
    plt.imshow(density + cv_image)
    plt.show()
    plt.imsave('1.png', density)
    return density


def get_file_list(file_path):
    return zip(os.listdir(file_path), [os.path.join(file_path, file) for file in os.listdir(file_path)])


if __name__ == '__main__':
    data_args = parse_args()
    for image_name, image_path in get_file_list(data_args.image_path):
        logger.info(image_name)
        cv_img = cv2.imread(image_path)
        # pil_img = Image.open(image_path)
        json_path = os.path.join(data_args.json_path, image_name.replace('.jpg', '.json'))
        if not os.path.exists(json_path):
            continue
        with open(json_path, 'r') as fr:
            json_points = json.load(open(json_path)).get('points')

        json_points = np.array(json_points, dtype=np.int)
        print(cv_img.shape)
        density_map = create_density(cv_img,json_points, cv_img.shape[0], cv_img.shape[1])
        np.save(os.path.join(data_args.npy_path, image_name.replace('.jpg', '.npy')), density_map)
