#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2020/5/18 下午5:31
# @Author  : Boyka
# @Email   : upcvagen@163.com
# @File    : convert_mat_to_json.py
# @Software: PyCharm
import argparse
import json
import logging
import os

import scipy.io as sci_io

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


def parse_args():
    parser = argparse.ArgumentParser(description="Path configuration")
    parser.add_argument(
        '--image_path',
        dest='image_path',
        help='',
        default='/home/chase/datasets/crowd_counting/ShanghaiTech/part_B_final/test_data/images',
        # default=None,
        type=str
    )
    parser.add_argument(
        '--mat_path',
        dest='mat_path',
        help='',
        default='/home/chase/datasets/crowd_counting/ShanghaiTech/part_B_final/test_data/ground_truth',
        # default=None,
        type=str
    )
    parser.add_argument(
        '--save_path',
        dest='save_path',
        help='',
        default='/home/chase/datasets/crowd_counting/ShanghaiTech/part_B_final/test_data/json',
        # default=None,
        type=str
    )
    return parser.parse_args()


def shanghai_crowd_points(mat_file_path):
    """
    :param mat_file_path: Mat file path
    :return: Numpy.ndarray of points
    """
    mat_info = sci_io.loadmat(mat_file_path)
    crowd_points = mat_info.get('image_info')[0][0][0][0][0]
    return crowd_points.tolist()


def shanghai_tech_annotation_convert(args):
    """

    :param args:
    :return:
    """
    if not os.path.exists(args.save_path):
        os.mkdir(args.save_path)
    for image in os.listdir(args.image_path):
        mat_file = os.path.join(args.mat_path, 'GT_' + image.replace('.jpg', '.mat'))
        if os.path.exists(mat_file):
            points = shanghai_crowd_points(mat_file)
            citation = "ZHANG, Yingying, et al. Single-image crowd counting via multi-column convolutional neural network. In: Proceedings of the IEEE conference on computer vision and pattern recognition. 2016. p. 589-597."
            info = {"image": image, "number": len(points), "points": points, "citation": citation}
            logger.info(image + " ————> " + str(len(points)))
            with open(os.path.join(args.save_path, image.replace('.jpg', '.json')), 'w') as fw:
                json.dump(info, fw)


if __name__ == '__main__':
    shanghat_tech_args = parse_args()
    shanghai_tech_annotation_convert(shanghat_tech_args)
