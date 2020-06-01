#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2020/6/1 下午3:45
# @Author  : Boyka
# @Email   : upcvagen@163.com
# @File    : dataset.py
# @Software: PyCharm

import os

import numpy as np
from PIL import Image


class CrowdCountingDataSet:
    def __init__(self, image_root, density_map_root, transforms):
        self.image_root = image_root
        self.density_map_root = density_map_root
        self.transforms = transforms
        self.images = os.listdir(image_root)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, image_name):
        image = Image.open(os.path.join(self.image_root, image_name)).convert("RGB")

        density_map = os.path.join(self.density_map_root)
        if not os.path.exists(density_map):
            density_map = np.zeros(image.size)
        else:
            density_map = np.load(density_map)

        if self.transforms is not None:
            image, density_map = self.transforms(image, density_map)

        return image, density_map
