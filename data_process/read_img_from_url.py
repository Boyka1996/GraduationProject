#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/3/24 22:25
# @Author  : Boyka
# @Email   : upcvagen@163.com
# @File    : read_img_from_url.py

import numpy as np
import urllib.request
import cv2


# URL到图片
def url_to_image(img_url):
    # download the image, convert it to a NumPy array, and then read
    # it into OpenCV format
    resp = urllib.request.urlopen(img_url)
    # bytearray将数据转换成（返回）一个新的字节数组
    # asarray 复制数据，将结构化数据转换成ndarray
    image = np.asarray(bytearray(resp.read()), dtype="uint8")
    # cv2.imdecode()函数将数据解码成Opencv图像格式
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    # return the image
    return image


# initialize the list of image URLs to download
urls = [
    "http://www.pyimagesearch.com/wp-content/uploads/2015/01/opencv_logo.png",
    "http://www.pyimagesearch.com/wp-content/uploads/2015/01/google_logo.png"
]

# loop over the image URLs
for url in urls:
    url_image = url_to_image(url)
    cv2.imshow("Image", url_image)
    cv2.waitKey(0)
