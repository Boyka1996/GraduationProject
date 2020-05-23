#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2020/5/19 下午2:54
# @Author  : Boyka
# @Email   : upcvagen@163.com
# @File    : draw.py
# @Software: PyCharm


import json

import matplotlib.image as mpimg  # mpimg 用于读取图片
import matplotlib.pyplot as plt  # plt 用于显示图片
import numpy as np

image = mpimg.imread('/home/chase/datasets/crowd_counting/NWPU/images/0001.jpg')  # 读取和代码处于同一目录下的 lena.png
json_file = '/home/chase/datasets/crowd_counting/NWPU/json/0001.json'
print(image.shape)  # (512, 512, 3)
with open(json_file, 'r') as fr:
    data = json.load(fr)
print(len(data.get('points')))
# print(data.get('points')[:, 0])
points = np.array(data.get('points'))
plt.imshow(image)  # 显示图片
plt.axis('off')  # 不显示坐标轴
plt.scatter(points[:, 0], points[:, 1], color=(1, 0, 0), s=3)
plt.show()