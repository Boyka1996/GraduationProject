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
        default='/home/chase/datasets/crowd_counting/ShanghaiTech/part_B_final/'
                'train_data/ShanghaiTechB_train_info_0.25.json',
        # default=None,
        type=str
    )

    return parser.parse_args()


def create_density(img, points, density_map_rows, density_map_cols):
    """

    :param img:
    :param points: (x,y) == (col_id,row_id)
    :param density_map_rows: height == rows
    :param density_map_cols: width == cols
    :return:
    """
    # Resize points and density map

    density_map_rows, density_map_cols = density_map_rows // 4, density_map_cols // 4
    points = np.array(points, dtype=np.float)[:, [1, 0]]
    points = points // 4
    img = cv2.resize(img, (density_map_cols, density_map_rows))

    density = np.zeros(shape=(density_map_rows, density_map_cols), dtype=np.float32)

    start_time = time.time()

    # Get the k-th nearest points
    neighborhoods = neighbors.NearestNeighbors(n_neighbors=5, algorithm='kd_tree', leaf_size=1200)
    neighborhoods.fit(points.copy())
    distances, neighborhood_id = neighborhoods.kneighbors()

    sigmas = distances.sum(axis=1) * 0.2 * 0.3
    points = np.floor(points).astype(np.int16)

    for i in range(len(points)):
        point = points[i]
        single_heat_map = np.zeros(shape=(density_map_rows, density_map_cols), dtype=np.float32)
        single_heat_map[min(point[0], density_map_rows - 1)][min(point[1], density_map_cols - 1)] = 1
        sigma = max(0.8, sigmas[i])
        radius = 50 * sigma

        # Scope of Gaussian kernel
        row_min = int(max(0, point[0] - radius))
        row_max = int(min(point[0] + radius, density_map_rows))
        col_min = int(max(0, point[1] - radius))
        col_max = int(min(point[1] + radius, density_map_cols))
        # ksize – Aperture size. It should be odd ( ksize mod 2 = 1 ) and positive.
        # sigma – Gaussian standard deviation.
        # If it is non-positive, it is computed from ksize as sigma = 0.3*((ksize-1)*0.5 - 1) + 0.8 .
        # ktype – Type of filter coefficients. It can be CV_32f or CV_64F
        # print(sigma)
        ksize = int(((sigma - 0.8) / 3 * 10 + 1)) * 2 + 1
        # print(ksize)
        single_heat_map = cv2.GaussianBlur(single_heat_map[row_min:row_max, col_min:col_max],
                                           ksize=(ksize, ksize), sigmaX=0)
        # plt.imshow(single_heat_map)
        # plt.show()
        density[row_min:row_max, col_min:col_max] += single_heat_map
    logger.info("***********************************")
    logger.info(time.time() - start_time)
    logger.info(len(points))
    logger.info(np.sum(density))
    # plt.imshow(density)
    # plt.show()
    # utils.show_desity_map(density)
    logger.info("***********************************")
    # print(density.shape, img.shape)
    # utils.show_density_image(density, img)
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
        cv_img = cv2.imread(image_path)
        json_path = os.path.join(data_args.json_path, image_name.replace('.jpg', '.json'))
        if not os.path.exists(json_path):
            continue
        with open(json_path, 'r') as fr:
            json_points = json.load(fr).get('points')
        plt.show()
        density_map = create_density(cv_img, np.array(json_points), cv_img.shape[0], cv_img.shape[1])
        # plt.imshow(density_map)
        # plt.show()

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
