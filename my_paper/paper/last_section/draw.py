#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time       : 2020/12/25 18:47
@Author     : Boyka
@Contact    : upcvagen@163.com
@File       : draw.py
@Project    : GraduationProject
@Description:
"""
import xml.etree.cElementTree as eT

import cv2
import numpy as np

def get_xml_objects(xml_path_):
    xml_annotations = []
    tree = eT.ElementTree(file=xml_path_)
    root = tree.getroot()
    for obj in root:
        if obj.tag == 'object':
            xmin = int(obj[4][0].text)
            ymin = int(obj[4][1].text)
            xmax = int(obj[4][2].text)
            ymax = int(obj[4][3].text)
            xml_annotations.append([xmin, ymin, xmax, ymax])

    return xml_annotations
def draw_iamge(img,points):
    for point in points:
        cv2.circle(img=img,center=(point[0],point[1]),radius=point[2],color=(0,0,255),thickness=1)

    # cv2.imshow("1",img)
    # cv2.waitKey(0)

image = "IMG_263.jpg"
anno = "IMG_263.xml"
cv_img = cv2.imread(image)
data = get_xml_objects(anno)
data=np.array(data)
# print(data)

# map(int,(data[:,2]+data[:,3])/3)
# points=list(zip(map(int,(data[:,0]+data[:,2])/2),map(int,(data[:,1]+data[:,3])/2)))


points=list(zip(map(int,(data[:,0]+data[:,2])/2),map(int,(data[:,1]+data[:,3])/2),map(int,(data[:,2]+data[:,3]-data[:,0]-data[:,1])*0.25)))
# print(type(np.array(points)[:,2]))
# print(np.array(points)[:,2])
points=np.array(points)
# print(np.sum(points[:,2])/len(data))
ave=int(np.sum(points[:,2])/len(points[:,2]))
points[:,2]=ave
# print(ave)
# for point in points:
#     cv2.rectangle(img=cv_img, pt1=(point[0], point[1]), pt2=(point[2], point[3]), color=(0, 0, 255), thickness=1)
# cv2.imshow('1',cv_img)
# cv2.waitKey(0)
draw_iamge(cv_img,points)
cv2.imwrite('result1.jpg',cv_img)