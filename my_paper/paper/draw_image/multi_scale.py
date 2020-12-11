#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time       : 2020/12/10 15:33
@Author     : Boyka
@Contact    : upcvagen@163.com
@File       : multi_scale.py
@Project    : GraduationProject
@Description:
"""
import cv2

if __name__ == '__main__':
    origin_image = "1.jpg"
    tar_image = "2.jpg"
    img = cv2.imread(origin_image)
    print(img.shape)
    cv2.imwrite(tar_image, img[0: int(img.shape[0] / 2), 0: int(img.shape[1] / 2)])
