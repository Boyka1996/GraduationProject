#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2020/5/19 上午11:04
# @Author  : Boyka
# @Email   : upcvagen@163.com
# @File    : convert_mat_to_json.py
# @Software: PyCharm
# A. B. Chan, Zhang-Sheng John Liang and N. Vasconcelos, "Privacy preserving crowd monitoring: Counting people without people models or tracking," 2008 IEEE Conference on Computer Vision and Pattern Recognition, Anchorage, AK, 2008, pp. 1-7, doi: 10.1109/CVPR.2008.4587569.

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
        default='/home/chase/datasets/crowd_counting/UCF_CC_50/images',
        # default=None,
        type=str
    )
    parser.add_argument(
        '--mat_path',
        dest='mat_path',
        help='',
        default='/home/chase/datasets/crowd_counting/UCF_CC_50/annotaion',
        # default=None,
        type=str
    )
    parser.add_argument(
        '--save_path',
        dest='save_path',
        help='',
        default='/home/chase/datasets/crowd_counting/UCF_CC_50/json',
        # default=None,
        type=str
    )
    return parser.parse_args()


def ucf_cc_50_crowd_points(mat_file_path):
    """
    :param mat_file_path: Mat file path
    :return: Numpy.ndarray of points
    """
    mat_info = sci_io.loadmat(mat_file_path)
    crowd_points = mat_info.get('annPoints')
    return crowd_points.tolist()


def ucf_cc_50_annotation_convert(args):
    """

    :param args:
    :return:
    """
    if not os.path.exists(args.save_path):
        os.mkdir(args.save_path)
    for image in os.listdir(args.image_path):
        mat_file = os.path.join(args.mat_path, image.replace('.jpg', '_ann.mat'))
        if os.path.exists(mat_file):
            points = ucf_cc_50_crowd_points(mat_file)
            citation = "IDREES, Haroon, et al. Multi-source multi-scale counting in extremely dense crowd images. In: Proceedings of the IEEE conference on computer vision and pattern recognition. 2013. p. 2547-2554."
            info = {"image": image, "number": len(points), "points": points, "citation": citation}
            logger.info(image + " ————> " + str(len(points)))
            with open(os.path.join(args.save_path, image.replace('.jpg', '.json')), 'w') as fw:
                json.dump(info, fw)


if __name__ == '__main__':
    ucf_cc_50_args = parse_args()
    ucf_cc_50_annotation_convert(ucf_cc_50_args)
