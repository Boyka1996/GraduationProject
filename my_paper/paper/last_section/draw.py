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
import h5py
import scipy.io as io
import PIL.Image as Image
import numpy as np
import os
import glob
from matplotlib import pyplot as plt
from scipy.ndimage.filters import gaussian_filter
import scipy
import scipy.spatial as spatial
import json
from matplotlib import cm as CM
from image import *
# from model import CSRNet
import torch
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


def gaussian_filter_density(gt):
    result=[]
    pts = np.array(gt)
    leafsize = 2048
    # build kdtree
    tree = scipy.spatial.KDTree(pts.copy(), leafsize=leafsize)
    # query kdtree
    distances, locations = tree.query(pts, k=4)

    print('generate density...')
    for i, pt in enumerate(pts):
        sigma = (distances[i][1] + distances[i][2] + distances[i][3]) * 0.1
        # pt2d = np.zeros(gt.shape, dtype=np.float32)
        # pt2d[pt[1], pt[0]] = 1.
        # if gt_count > 1:
        #     sigma = (distances[i][1] + distances[i][2] + distances[i][3]) * 0.1
        # else:
        #     sigma = np.average(np.array(gt.shape)) / 2. / 2.  # case: 1 point
        print(sigma)
        result.append([pt[0],pt[1],int(sigma)])
        # density += scipy.ndimage.filters.gaussian_filter(pt2d, sigma, mode='constant')
    print('done.')
    return result
    # return density


def draw_iamge(img, points):
    for point in points:
        cv2.circle(img=img, center=(point[0], point[1]), radius=point[2], color=(0, 0, 255), thickness=2)

    # cv2.imshow("1",img)
    # cv2.waitKey(0)


image = "IMG_263.jpg"
anno = "IMG_263.xml"
cv_img = cv2.imread(image)
data = get_xml_objects(anno)
data = np.array(data)

points = list(zip(map(int, (data[:, 0] + data[:, 2]) / 2), map(int, (data[:, 1] + data[:, 3]) / 2)))
points=gaussian_filter_density(points)
print(points)
# points=list(zip(map(int,(data[:,0]+data[:,2])/2),map(int,(data[:,1]+data[:,3])/2),map(int,(data[:,2]+data[:,3]-data[:,0]-data[:,1])*0.275)))


# points=list(zip(map(int,(data[:,0]+data[:,2])/2),map(int,(data[:,1]+data[:,3])/2),map(int,(data[:,2]+data[:,3]-data[:,0]-data[:,1])*0.25)))
## points=np.array(points)
# ave=int(np.sum(points[:,2])/len(points[:,2]))
# points[:,2]=ave

draw_iamge(cv_img, points)
cv2.imwrite('result2.jpg', cv_img)
