#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time       : 2020/12/23 23:45
@Author     : Boyka
@Contact    : upcvagen@163.com
@File       : draw1.py
@Project    : GraduationProject
@Description:
"""
import pandas
import numpy as np
import cv2
def draw_iamge(img,points):
    for point in points:
        cv2.circle(img=img,center=(point[0],point[1]),radius=point[2],color=(0,0,255),thickness=1)
    cv2.imshow("1",img)
    cv2.waitKey(0)
data = pandas.read_csv('4049.txt',encoding='gb2312',delim_whitespace=True)
# data = pandas.read_csv('4049.txt')
# print(len(np.array(data)))
data=np.array(data)
print(data)
points=list(zip(data[:,0],data[:,1],map(int,(data[:,2]+data[:,3])/3)))
# points=list(zip(map(int,data[:,0]/2+data[:,1]/2),map(int,data[:,2]/2+data[:,3]/2),map(int,(data[:,0]+data[:,1]+data[:,2]+data[:,3])/4)))
print(points)
draw_iamge(cv2.imread('4049.jpg'),points)