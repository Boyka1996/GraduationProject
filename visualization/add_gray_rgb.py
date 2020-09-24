#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2020/9/21 上午9:53
# @Author  : Boyka
# @Email   : upcvagen@163.com
# @Software: PyCharm
import logging

import cv2
import numpy as np

LOGGER = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

RGB_PATH = '/home/chase/datasets/crowd_counting/ShanghaiTech/part_A_final/train_data/images/IMG_1.jpg'
GRAY_PATH = '/home/chase/datasets/crowd_counting/ShanghaiTech/part_A_final/train_data/npy_0.25/IMG_1.npy'


def add_gray_rgb(gray, rgb):
    LOGGER.info(rgb.shape)
    LOGGER.info(type(rgb[0][0][0]))
    gray = 255 * gray
    gray = np.array(gray, dtype='uint8')
    gray = cv2.resize(gray, (0, 0), fx=4, fy=4, interpolation=cv2.INTER_NEAREST)
    blank = np.zeros(shape=gray.shape, dtype='uint8')
    # print(gray.shape)
    # print(blank.shape)
    # blank = np.stack((blank, gray))
    print(blank.shape)
    gray = np.stack((blank, blank, gray), axis=2)

    # gray=np.stack((gray,blank,blank),axis=2)
    print(gray.shape)

    # cv2.imshow('1', gray)
    # cv2.waitKey()
    # gray = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    print(gray.shape)
    # gray[::, 2] = 0
    # gray[::, 1] = 0
    cv2.imshow('1', gray)
    cv2.waitKey()

    LOGGER.info(rgb.shape)
    LOGGER.info(gray.shape)
    result = cv2.addWeighted(gray, 1, rgb, 0.5, 0)
    cv2.imshow('1', result)
    cv2.waitKey()
    return result


if __name__ == '__main__':
    add_gray_rgb(np.load(GRAY_PATH), cv2.imread(RGB_PATH))
