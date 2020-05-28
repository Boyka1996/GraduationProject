#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2020/5/23 上午8:12
# @Author  : Boyka
# @Email   : upcvagen@163.com
# @File    : heat_map_survey.py
# @Software: PyCharm


# -*-encoding: utf-8 -*-

import argparse
import json
import logging
import os

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy.ndimage import filters

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


def parse_args():
    parser = argparse.ArgumentParser(description="Path configuration")
    parser.add_argument(
        '--image_path',
        dest='image_path',
        help='',
        default='/home/chase/datasets/crowd_counting/UCF-QNRF_ECCV18/Train/images',
        # default=None,
        type=str
    )

    parser.add_argument(
        '--json_path',
        dest='json_path',
        help='',
        default='/home/chase/datasets/crowd_counting/UCF-QNRF_ECCV18/Train/json',
        # default=None,
        type=str
    )
    parser.add_argument(
        '--npy_path',
        dest='npy_path',
        help='',
        default='/home/chase/datasets/crowd_counting/UCF-QNRF_ECCV18/Train/npy',
        # default=None,
        type=str
    )
    return parser.parse_args()


# gauss kernel
def gen_gauss_kernels(kernel_size=15, sigma=4):
    kernel_shape = (kernel_size, kernel_size)
    kernel_center = (kernel_size // 2, kernel_size // 2)

    arr = np.zeros(kernel_shape).astype(float)
    arr[kernel_center] = 1

    arr = filters.gaussian_filter(arr, sigma, mode='constant')
    kernel = arr / arr.sum()
    return kernel


def gaussian_filter_density(non_zero_points, map_h, map_w):
    """
    Fast gaussian filter implementation : using precomputed distances and kernels
    """
    gt_count = non_zero_points.shape[0]
    density_map = np.zeros((map_h, map_w), dtype=np.float32)

    for i in range(gt_count):
        point_y, point_x = non_zero_points[i]
        # print(point_x, point_y)
        kernel_size = 15 // 2
        kernel = gen_gauss_kernels(kernel_size * 2 + 1, 4)
        min_img_x = int(max(0, point_x - kernel_size))
        min_img_y = int(max(0, point_y - kernel_size))
        max_img_x = int(min(point_x + kernel_size + 1, map_h - 1))
        max_img_y = int(min(point_y + kernel_size + 1, map_w - 1))
        # print(min_img_x, min_img_y, max_img_x, max_img_y)
        kernel_x_min = int(kernel_size - point_x if point_x <= kernel_size else 0)
        kernel_y_min = int(kernel_size - point_y if point_y <= kernel_size else 0)
        kernel_x_max = int(kernel_x_min + max_img_x - min_img_x)
        kernel_y_max = int(kernel_y_min + max_img_y - min_img_y)
        # print(kernel_x_max, kernel_x_min, kernel_y_max, kernel_y_min)

        density_map[min_img_x:max_img_x, min_img_y:max_img_y] += kernel[kernel_x_min:kernel_x_max,
                                                                 kernel_y_min:kernel_y_max]
    fig1 = plt.figure('fig1')
    plt.imshow(density_map)
    plt.show()
    return density_map


def get_file_list(file_path):
    return zip(os.listdir(file_path), [os.path.join(file_path, file) for file in os.listdir(file_path)])


if __name__ == '__main__':

    data_args = parse_args()
    mod = 16
    scale = 1024
    # name_list,path_list= get_file_list(data_args.image_path)
    for image_name, image_path in get_file_list(data_args.image_path):
        pil_img = Image.open(image_path)
        width, height = pil_img.size
        json_path = os.path.join(data_args.json_path, image_name.replace('.jpg', '.json'))
        with open(json_path, 'r') as fr:
            points = json.load(open(json_path)).get('points')
        points = np.array(points)
        if max(width, height) > scale:
            if width == max(width, height):
                nw, nh = scale, round(height * scale / width / mod) * mod
            else:
                nh, nw = scale, round(width * scale / height / mod) * mod
        else:
            nw, nh = round((width / mod)) * mod, round((height / mod)) * mod
        pil_img.resize((nw, nh), Image.BILINEAR)
        #
        # if len(points) > 0:
        #     points[:, 0] = points[:, 0].clip(0, width - 1)
        #     points[:, 1] = points[:, 1].clip(0, height - 1)
        #     points[:, 0] = (points[:, 0] / width * nw).round().astype(int)
        #     points[:, 1] = (points[:, 1] / height * nh).round().astype(int)
        den = gaussian_filter_density(points, nh, nw)
