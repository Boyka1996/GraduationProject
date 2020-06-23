#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2020/5/23 上午8:50
# @Author  : Boyka
# @Email   : upcvagen@163.com
# @File    : resize_img.py
# @Software: PyCharm

import os
from PIL import Image
from PIL.Image import NEAREST, BILINEAR, BICUBIC, LANCZOS, BOX, HAMMING

resmaple_list = [NEAREST, BILINEAR, BICUBIC, LANCZOS, BOX, HAMMING]
path = r'/home/chase/datasets/crowd_counting/ShanghaiTech/part_A_final/test_data/images/'

filename = 'IMG_2.jpg'
fullname = os.path.join(path, filename)

image = Image.open(fullname)
cut_size = [int(x * 0.1) for x in image.size]
new_size = (cut_size[0] * 3, cut_size[1] * 2)

new_image = Image.new('RGB', new_size)
for i in range(0, 3):
    im = image.resize(cut_size, resample=resmaple_list[i])
    new_image.paste(im, (i * cut_size[0], 0))

for i in range(3, 6):
    im = image.resize(cut_size, resample=resmaple_list[i])
    new_image.paste(im, ((i - 3) * cut_size[0], cut_size[1]))

new_image.show()
new_image.save(os.path.join('/home/chase/datasets/crowd_counting/UCF_CC_50', 'outputImage.jpg'))