#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2020/5/28 上午8:09
# @Author  : Boyka
# @Email   : upcvagen@163.com
# @File    : cv2_image_segmentation.py
# @Software: PyCharm

import numpy as np
import cv2
np.set_printoptions(threshold=np.inf)
img = cv2.imread('/home/chase/datasets/crowd_counting/NWPU/images/3100.jpg',1)
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

cv2.imshow('thresh', thresh)
kernel = np.ones((3,3),np.uint8)
# 开运算
opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 2)
cv2.imshow('opening', opening)
# sure background area   膨胀后确保背景
sure_bg = cv2.dilate(opening,kernel,iterations=3)
cv2.imshow('sure_bg', sure_bg)
# 第二个参数0,1,2 分别表示CV_DIST_L1, CV_DIST_L2 , CV_DIST_C
dist_transform = cv2.distanceTransform(opening,1,5)
ret, sure_fg = cv2.threshold(dist_transform,0.7*dist_transform.max(),255,0)
cv2.imshow('sure_fg', sure_fg)
# Finding unknown region
sure_fg = np.uint8(sure_fg)
unknown = cv2.subtract(sure_bg,sure_fg)
cv2.imshow('unknown', unknown)
# Marker labelling
# 求取连通域
ret, markers1 = cv2.connectedComponents(sure_fg)
# Add one to all labels so that sure background is not 0, but 1
markers = markers1+1
# Now, mark the region of unknown with zero
markers[unknown==255] = 0
markers3 = cv2.watershed(img,markers)
img[markers3 == -1] = [255,255,0]
cv2.imshow("img",img)
cv2.waitKey(0)
cv2.destroyAllWindows()
