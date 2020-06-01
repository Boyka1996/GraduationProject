#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2020/5/28 下午3:11
# @Author  : Boyka
# @Email   : upcvagen@163.com
# @File    : shanghai.py
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
        default='/home/chase/datasets/crowd_counting/ShanghaiTech/part_A_final/train_data/images',
        # default=None,
        type=str
    )

    parser.add_argument(
        '--json_path',
        dest='json_path',
        help='',
        default='/home/chase/datasets/crowd_counting/ShanghaiTech/part_A_final/train_data/json',
        # default=None,
        type=str
    )
    parser.add_argument(
        '--npy_path',
        dest='npy_path',
        help='',
        default='/home/chase/datasets/crowd_counting/ShanghaiTech/part_A_final/train_data/npy',
        # default=None,
        type=str
    )
    parser.add_argument(
        '--data_set_info',
        dest='data_set_info',
        help='',
        default='/home/chase/datasets/crowd_counting/ShanghaiTech/part_A_final/train_data/shanghai_tech_a_train_info.json',
        # default=None,
        type=str
    )
    return parser.parse_args()


def create_density(points, density_map_rows, density_map_cols):
    """

    :param points: (x,y) == (col_id,row_id)
    :param density_map_rows: height == rows
    :param density_map_cols: width == cols
    :return:
    """
    density = np.zeros(shape=(density_map_rows, density_map_cols), dtype=np.float32)
    # if len(points) < 5:
    #     return density
    points = points[:, [1, 0]]
    start_time = time.time()
    neighborhoods = neighbors.NearestNeighbors(n_neighbors=5, algorithm='kd_tree', leaf_size=10000)
    neighborhoods.fit(points.copy())
    distances, _ = neighborhoods.kneighbors()

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
        radius = 5 * sigma
        row_min = max(0, point[0] - radius)
        row_max = min(point[0] + radius, density_map_rows)
        col_min = max(0, point[1] - radius)
        col_max = min(point[1] + radius, density_map_cols)
        single_heat_map = cv2.GaussianBlur(single_heat_map[row_min:row_max, col_min:col_max],
                                           ksize=(3 * sigma, 3 * sigma), sigmaX=sigma, sigmaY=sigma)
        density[row_min:row_max, col_min:col_max] += single_heat_map
    logger.info("***********************************")
    logger.info(time.time() - start_time)
    logger.info(len(points))
    logger.info(np.sum(density))

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
        # cv_img = Image.open(image_path)
        cv_img = cv2.imread(image_path)
        json_path = os.path.join(data_args.json_path, image_name.replace('.jpg', '.json'))
        if not os.path.exists(json_path):
            continue
        with open(json_path, 'r') as fr:
            json_points = json.load(open(json_path)).get('points')
        json_points = np.array(json_points, dtype=np.int)
        if image_name == '1015.jpg':
            json_points = json_points[:, [1, 0]]
        # Because the coordinates of '1015.jpg' in NWPU data set is transposed
        print(cv_img.shape)
        density_map = create_density(json_points, cv_img.shape[0], cv_img.shape[1])

        # np.save(os.path.join(data_args.npy_path, image_name.replace('.jpg', '.npy')), density_map)

        # show_img = cv2.cvtColor(density_map*255, cv2.COLOR_GRAY2RGB)
        # print(show_img.shape)
        # dst = cv2.addWeighted(cv_img, 0.9, show_img, 0.1,0)
        # for i in range(len(cv_img)):
        #     for j in range(len(cv_img[i])):
        #         cv_img[i][j] = cv_img[i][j] * density_map[i][j]*(1/max(density_map))
        # cv2.imshow('1', cv_img)
        # cv2.waitKey()
        # # plt.imshow(show_img)
        # plt.show()

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

        with open('mall_info.json', 'w') as fw:
            json.dump(data_set_info, fw)
