#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2020/5/19 上午9:52
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
        default='/home/chase/datasets/crowd_counting/mall/frames',
        # default=None,
        type=str
    )
    parser.add_argument(
        '--mat_path',
        dest='mat_path',
        help='',
        default='/home/chase/datasets/crowd_counting/mall/mall_gt.mat',
        # default=None,
        type=str
    )
    parser.add_argument(
        '--save_path',
        dest='save_path',
        help='',
        default='/home/chase/datasets/crowd_counting/mall/json',
        # default=None,
        type=str
    )
    return parser.parse_args()


def mall_crowd_points(mat_file_path):
    """
    :param mat_file_path: Mat file path
    :return: Numpy.ndarray of points
    """
    mat_info = sci_io.loadmat(mat_file_path)
    crowd_points = mat_info.get('frame')
    return crowd_points.tolist()[0]


def mall_annotation_convert(args):
    """

    :param args:
    :return:
    """
    if not os.path.exists(args.save_path):
        os.mkdir(args.save_path)
    image_points = mall_crowd_points(args.mat_path)
    print(len(image_points))
    for points_id, points in enumerate(image_points):
        image = 'seq_' + '0' * (6 - len(str(points_id + 1))) + str(points_id + 1) + '.jpg'
        if not os.path.exists(os.path.join(args.image_path, image)):
            continue
        points = points[0][0][0].tolist()

        citation = "CHEN, Ke, et al. Feature mining for localised crowd counting. In: BMVC. 2012. p. 3."
        info = {"image": image, "number": len(points), "points": points, "citation": citation}
        logger.info(image + " ————> " + str(len(points)))
        with open(os.path.join(args.save_path, image.replace('.jpg', '.json')), 'w') as fw:
            json.dump(info, fw)


if __name__ == '__main__':
    mall_args = parse_args()
    mall_annotation_convert(mall_args)
    # mat_info = sci_io.loadmat(mall_args.mat_path)
    # logger.info(points)
    # logger.info(points.shape)
    # perspective = \
    # sci_io.loadmat('/home/chase/datasets/crowd_counting/mall_dataset/mall_dataset/perspective_roi.mat').get("roi")[0][0]
    # feat = sci_io.loadmat('/home/chase/datasets/crowd_counting/mall_dataset/mall_dataset/mall_feat.mat').get('x')
    # crowd_points = mat_info.get('x')
    # logger.info(crowd_points)
    # print()
