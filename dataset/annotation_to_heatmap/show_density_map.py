#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time       : 11/19/20 6:09 PM
# @Author     : Boyka
# @Contact    : upcvagen@163.com
# @File       : show_density_map.py
# @Project    : GraduationProject
# @Description:
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import json
"""

"""
# plt.imshow(Image.open('/home/chase/datasets/crowd_counting/ShanghaiTech/part_A_final/train_data/images/IMG_1.jpg'))
# plt.show()
gt_file = np.load('/home/chase/datasets/crowd_counting/ShanghaiTech/part_A_final/train_data/density_map/IMG_1.npy')
plt.imshow(gt_file,cmap=plt.cm.jet)
# plt.imshow(gt_file)

plt.show()


print(np.sum(gt_file))
with open('/home/chase/datasets/crowd_counting/ShanghaiTech/part_A_final/train_data/json/IMG_1.json', 'r') as fr:
    json_points = json.load(fr).get('points')
print(len(json_points))
