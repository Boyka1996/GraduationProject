#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time       : 2020/12/23 22:53
@Author     : Boyka
@Contact    : upcvagen@163.com
@File       : draw.py
@Project    : GraduationProject
@Description:
"""
import json
import logging
import os
import xml.etree.cElementTree as eT

import cv2


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


image = "PartA_00208.jpg"
anno = "PartA_00208.xml"
cv_img = cv2.imread(image)
points = get_xml_objects(anno)
print(points)
for point in points:
    cv2.rectangle(img=cv_img, pt1=(point[0], point[1]), pt2=(point[2], point[3]), color=(0, 0, 255), thickness=1)
# cv2.imshow('1',cv_img)
# cv2.waitKey(0)
cv2.imwrite('result.jpg',cv_img)