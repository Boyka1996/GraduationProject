#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2020/7/4 下午4:54
# @Author  : Boyka
# @Email   : upcvagen@163.com
# @Software: PyCharm


import argparse
import json
import logging
import math
import os
import time

import cv2
import matplotlib.pyplot as plt
import numpy as np
import scipy
from PIL import Image
from sklearn import neighbors

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


def parse_args():
    parser = argparse.ArgumentParser(description="Path configuration")
    parser.add_argument(
        '--image_path',
        dest='image_path',
        help='',
        default='/home/chase/datasets/crowd_counting/ShanghaiTech/part_B_final/train_data/images',
        # default=None,
        type=str
    )

    parser.add_argument(
        '--json_path',
        dest='json_path',
        help='',
        default='/home/chase/datasets/crowd_counting/ShanghaiTech/part_B_final/train_data/json',
        # default=None,
        type=str
    )
    parser.add_argument(
        '--npy_path',
        dest='npy_path',
        help='',
        default='/home/chase/datasets/crowd_counting/ShanghaiTech/part_B_final/train_data/npy_0.25',
        # default=None,
        type=str
    ),
    parser.add_argument(
        '--data_set_info',
        dest='data_set_info',
        help='',
        default='/home/chase/datasets/crowd_counting/ShanghaiTech/part_B_final/train_data/ShanghaiTechB_train_info_0.25.json',
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

    arr = scipy.ndimage.filters.gaussian_filter(arr, sigma, mode='constant')
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
    return density_map


def create_density(points, density_map_rows, density_map_cols):
    """

    :param points: (x,y) == (col_id,row_id)
    :param density_map_rows: height == rows
    :param density_map_cols: width == cols
    :return:
    """
    # print(density_map_rows, density_map_cols )
    density_map_rows, density_map_cols = density_map_rows // 4, density_map_cols // 4

    points = np.array(points, dtype=np.float)[:, [1, 0]]
    points = points // 4

    # points = np.array(points, dtype=np.float)[:, [1, 0]]

    density = np.zeros(shape=(density_map_rows, density_map_cols), dtype=np.float32)

    start_time = time.time()

    neighborhoods = neighbors.NearestNeighbors(n_neighbors=4, algorithm='kd_tree', leaf_size=1200)
    neighborhoods.fit(points.copy())
    distances, neighborhood_id = neighborhoods.kneighbors()

    sigmas = distances.sum(axis=1) * 0.75
    points = np.floor(points).astype(np.int16)
    # points = points.astype(np.int16)
    # sigmas=np.ceil(sigmas)
    # print(density_map_rows, density_map_cols )
    for i in range(len(points)):

        point = points[i]
        single_heat_map = np.zeros(shape=(density_map_rows, density_map_cols), dtype=np.float32)
        plt.imshow(single_heat_map)
        plt.show()
        single_heat_map[min(point[0], density_map_rows - 1)][min(point[1], density_map_cols - 1)] = 1
        plt.imshow(single_heat_map)
        plt.show()
        sigma = int(sigmas[i])
        # Scale adaptive Gaussian kernel
        if sigma % 2 == 0:
            sigma += 1
            # print(sigma)
        radius = 5 * sigma

        # Scope of Gaussian kernel
        row_min = max(0, point[0] - radius)
        row_max = min(point[0] + radius, density_map_rows)
        col_min = max(0, point[1] - radius)
        col_max = min(point[1] + radius, density_map_cols)

        single_heat_map = cv2.GaussianBlur(single_heat_map[row_min:row_max, col_min:col_max],
                                           ksize=(3 * sigma, 3 * sigma), sigmaX=sigma, sigmaY=sigma)
        plt.imshow(single_heat_map)
        plt.show()
        density[row_min:row_max, col_min:col_max] += single_heat_map[row_min:row_max, col_min:col_max]
    logger.info("***********************************")
    logger.info(time.time() - start_time)
    logger.info(len(points))
    logger.info(np.sum(density))
    plt.imshow(density)
    plt.show()
    logger.info("***********************************")
    return density


def get_file_list(file_path):
    return zip(os.listdir(file_path), [os.path.join(file_path, file) for file in os.listdir(file_path)])


if __name__ == '__main__':
    data_args = parse_args()
    if not os.path.exists(data_args.npy_path):
        os.makedirs(data_args.npy_path)
    ground_truth_info = dict()
    density_map_person_num = dict()
    detailed_density_map_person_num = dict()
    density_map_person_num_error = dict()
    detailed_density_map_person_num_error = dict()
    data_set_info = {'ground_truth_info': ground_truth_info, 'density_map_person_num': density_map_person_num,
                     'detailed_density_map_person_num': detailed_density_map_person_num,
                     'density_map_person_num_error': density_map_person_num_error,
                     'detailed_density_map_person_num_error': detailed_density_map_person_num_error}

    for image_name, image_path in get_file_list(data_args.image_path):
        logger.info(image_name)
        pil_img = Image.open(image_path).convert("RGB")
        cv_img = cv2.imread(image_path)
        json_path = os.path.join(data_args.json_path, image_name.replace('.jpg', '.json'))
        if not os.path.exists(json_path):
            continue
        with open(json_path, 'r') as fr:
            json_points = json.load(fr).get('points')
        plt.show()
        # json_points = np.array(json_points)
        # density_map=gaussian_filter_density(json_points, pil_img.size[1], pil_img.size[0])
        density_map = create_density(np.array(json_points), pil_img.size[1], pil_img.size[0])
        plt.imshow(density_map)
        plt.show()

        np.save(os.path.join(data_args.npy_path, image_name.replace('.jpg', '.npy')), density_map)

        detailed_person_num = float(np.sum(density_map))
        person_num = int(detailed_person_num)

        ground_truth_info[image_name.replace('.jpg', '')] = len(json_points)
        density_map_person_num[image_name.replace('.jpg', '')] = person_num
        detailed_density_map_person_num[image_name.replace('.jpg', '')] = detailed_person_num
        density_map_person_num_error[image_name.replace('.jpg', '')] = int(
            math.fabs(detailed_person_num - len(json_points)))
        detailed_density_map_person_num_error[image_name.replace('.jpg', '')] = math.fabs(
            detailed_person_num - len(json_points))

    total_error = int(np.sum(list(density_map_person_num_error.values())))
    total_detailed_error = float(np.sum(list(detailed_density_map_person_num_error.values())))

    print(total_error)
    print(total_error / len(density_map_person_num_error.values()))
    print(total_detailed_error)
    print(total_detailed_error / len(density_map_person_num_error.values()))

    data_set_info['total_error'] = total_error
    data_set_info['total_detailed_error'] = total_detailed_error
    data_set_info['mean_error'] = total_error / len(density_map_person_num_error.values())
    data_set_info['mean_detailed_error'] = total_detailed_error / len(density_map_person_num_error.values())

    with open(data_args.data_set_info, 'w') as fw:
        json.dump(data_set_info, fw)
