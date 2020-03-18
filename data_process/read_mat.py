# -*- coding: utf-8 -*-
# School                 ：UPC
# Author                 ：Boyka
# E-mail                 ：upcvagen@163.com
# File Name              ：read_mat.py
# Computer User          ：Administrator 
# Current Project        ：GraduationProject
# Development Time       ：2020/3/17  1:06 
# Development Tool       ：PyCharm

import scipy.io as sci_io


def shanghai_crowd_points(mat_file_path):
    """
    :param mat_file_path: Mat file path
    :return: Numpy.ndarray of points with shape of (n,2)
    """
    mat_info = sci_io.loadmat(mat_file_path)
    crowd_points = mat_info.get('image_info')[0][0][0][0][0]
    return crowd_points


if __name__ == '__main__':
    data_path = 'GT_IMG_1.mat'
    points = shanghai_crowd_points(data_path)
    print(type(points[0]))
    print(points[0][0])
    print(len(points[0]))
