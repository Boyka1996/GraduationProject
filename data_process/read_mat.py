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
import numpy as np

if __name__ == '__main__':
    data_path = 'GT_IMG_1.mat'
    data = sci_io.loadmat(data_path)
    data = data['image_info']
    # data=np.array(data['image_info'])
    points = data[0][0][0][0][0]
    print(len(data[0][0][0][0][0]))
