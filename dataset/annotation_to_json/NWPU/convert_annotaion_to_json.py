#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2020/5/19 下午2:24
# @Author  : Boyka
# @Email   : upcvagen@163.com
# @File    : convert_annotaion_to_json.py
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
        default='/home/chase/datasets/crowd_counting/NWPU/images',
        # default=None,
        type=str
    )
    parser.add_argument(
        '--mat_path',
        dest='mat_path',
        help='',
        default='/home/chase/datasets/crowd_counting/NWPU/mats',
        # default=None,
        type=str
    )
    parser.add_argument(
        '--save_path',
        dest='save_path',
        help='',
        default='/home/chase/datasets/crowd_counting/NWPU/json',
        # default=None,
        type=str
    )
    return parser.parse_args()


def nwpu_crowd_points(mat_file_path):
    """
    :param mat_file_path: Mat file path
    :return: Numpy.ndarray of points
    """
    mat_info = sci_io.loadmat(mat_file_path)
    crowd_points = mat_info.get('annPoints')
    return crowd_points.tolist()


def nwpu_annotation_convert(args):
    """

    :param args:
    :return:
    """
    if not os.path.exists(args.save_path):
        os.mkdir(args.save_path)
    for image in os.listdir(args.image_path):
        mat_file = os.path.join(args.mat_path, image.replace('.jpg', '.mat'))
        if os.path.exists(mat_file):
            points = nwpu_crowd_points(mat_file)
            citation = "Wang, Qi & Gao, Junyu & Lin, Wei & Li, Xuelong. (2020). NWPU-Crowd: A Large-Scale Benchmark for Crowd Counting."
            info = {"image": image, "number": len(points), "points": points, "citation": citation}
            logger.info(image + " ————> " + str(len(points)))
            with open(os.path.join(args.save_path, image.replace('.jpg', '.json')), 'w') as fw:
                json.dump(info, fw)


if __name__ == '__main__':
    nwpu_args = parse_args()
    nwpu_annotation_convert(nwpu_args)
