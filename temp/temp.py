#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time       : 2021/1/7 13:59
@Author     : Boyka
@Contact    : upcvagen@163.com
@File       : temp.py
@Project    : GraduationProject
@Description:
"""
import cv2
img=cv2.imread('1.jpg')
print(img.shape)
img=img[300:img.shape[0]-300]
print(img.shape)
out=cv2.resize(img,(150,200))
cv2.imwrite('2.jpg',out)
cv2.imshow('1',out)
cv2.waitKey()
print(out.shape)