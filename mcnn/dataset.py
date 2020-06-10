#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2020/6/10 上午11:32
# @Author  : Boyka
# @Email   : upcvagen@163.com
# @File    : dataset.py
# @Software: PyCharm
import os

import numpy as np
from PIL import Image


class ShanghaiTechA:
    def __init__(self, root, transforms):
        self.root = root
        self.transforms = transforms
        self.images = list(sorted(os.listdir(os.path.join(root, "images"))))
        self.gt_density_maps = list(os.listdir(os.path.join(root, "npy")))

    def __getitem__(self, item):
        image = Image.open(os.path.join(self.root, "images", self.images[item])).convert("RGB")
        gt_density_map_path = os.path.join(self.root, "images", self.gt_density_maps[item])
        if not os.path.exists(gt_density_map_path):
            gt_density_map = np.zeros(shape=image.size())
        else:
            gt_density_map = np.load(gt_density_map_path)
        if self.transforms is not None:
            image, gt_density_map = self.transforms(image, gt_density_map)
        return image, gt_density_map

    def __len__(self):
        return len(self.images)
