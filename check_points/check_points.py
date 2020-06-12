#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2020/6/10 下午5:44
# @Author  : Boyka
# @Email   : upcvagen@163.com
# @File    : check_points.py
# @Software: PyCharm

import json

import cv2
import matplotlib.pyplot as plt
import numpy as np
if __name__ == '__main__':
    with open('IMG_26.json', 'r') as fr:
        json_points = json.load(fr).get('points')
    show_img = cv2.imread('IMG_26.jpg')
    json_points = np.array(json_points, dtype=np.int)
    # json_points = json_points[:, [1, 0]]
    tmp=np.zeros(show_img.shape)
    tmp[100][10]=[255,255,255]
    cv2.circle(tmp, (100, 10), 3, (0, 0, 255), -1)
    cv2.imwrite('null.jpg',tmp)
    for point in json_points:
        print(point)
        cv2.circle(show_img, (point[0],point[1]),1,(255, 0, 0),-1)


    cv2.imshow('1',show_img)
    cv2.waitKey()
    # plt.imshow(show_img)
    # plt.plot(json_points[:, 0], json_points[:, 1], "ro")
    # plt.show()
