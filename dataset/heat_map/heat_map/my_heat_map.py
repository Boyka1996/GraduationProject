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
import math
import os
import time

import cv2
import numpy as np
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
    ),
    parser.add_argument(
        '--data_set_info',
        dest='data_set_info',
        help='',
        default='/home/chase/datasets/crowd_counting/UCF-QNRF_ECCV18/Train/qurf_train_info.json',
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
    start_time = time.time()

    neighborhoods = neighbors.NearestNeighbors(n_neighbors=4, algorithm='kd_tree', leaf_size=1200)
    neighborhoods.fit(points.copy())
    distances, neighborhood_id = neighborhoods.kneighbors()

    sigmas = distances.sum(axis=1) * 0.075
    for i in range(len(points)):
        point = points[i]
        single_heat_map = np.zeros(shape=(density_map_rows, density_map_cols), dtype=np.float32)
        single_heat_map[point[0]][point[1]] = 1
        # sigmas[i] = int(sigmas[i])
        sigma = sigmas[i].astype(np.int16)
        # Scale adaptive Gaussian kernel
        if sigma == 0:
            near = neighborhood_id[i]
            for fk_id in near:
                print(points[fk_id])
        if sigma % 2 == 0:
            sigma += 1
        # radius = 100 * sigma
        radius = 5 * sigma
        # print(sigma)
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
    # plt.imshow(density)
    # plt.show()
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
        json_path = os.path.join(data_args.json_path, image_name.replace('.jpg', '.json'))
        if not os.path.exists(json_path):
            continue
        with open(json_path, 'r') as fr:
            json_points = json.load(fr).get('points')
        json_points = np.array(json_points, dtype=np.int)[:, [1, 0]]
        # json_points[:, 0] = json_points[:, 0].clip(0, cv_img.shape[0] - 1)
        # json_points[:, 1] = json_points[:, 1].clip(0, cv_img.shape[1] - 1)
        # if image_name == '1015.jpg':
        #     json_points = json_points[:, [1, 0]]
        print(np.array(pil_img).shape)
        print(type(pil_img))
        print(pil_img.size)
        print(len(json_points))
        # plt.imshow(cv_img[:, :, 0])
        # plt.imshow(cv_img[:, :, 1])
        # plt.imshow(cv_img[:, :, 2])
        # plt.show()
        # density_map = create_density(json_points, cv_img.shape[0], cv_img.shape[1])
        density_map = create_density(json_points, pil_img.size[0], pil_img.size[1])

        np.save(os.path.join(data_args.npy_path, image_name.replace('.jpg', '.npy')), density_map)

        # plt.imshow(show_img)
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
