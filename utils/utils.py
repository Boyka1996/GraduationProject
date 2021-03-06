#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2020/7/4 下午2:19
# @Author  : Boyka
# @Email   : upcvagen@163.com
# @Software: PyCharm
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np


def show_density_image(density_map, image):
    density_map = 255.0 * density_map
    density_map = np.stack((density_map, density_map, density_map), axis=0)

    density_map = np.swapaxes(density_map, 0, 1)
    density_map = np.swapaxes(density_map, 1, 2)

    result = image*0.5 + density_map*0.5

    # result=cv2.addWeighted(image,0.5,density_map,0.5,0)
    # cv2.imshow('1', density_map)
    # cv2.waitKey()
    image=255 * image / np.max(image)
    plt.imshow(density_map)
    plt.show()


def show_desity_map(density_map):
    density_map = 1000 * density_map / np.max(density_map)
    plt.imshow(density_map)
    plt.savefig('1.png')
    plt.show()
    # cv2.imshow('',density_map)
    # cv2.waitKey()


def save_results(input_img, gt_data, density_map, output_dir, fname='results.png'):
    input_img = input_img[0][0]
    gt_data = 255 * gt_data / np.max(gt_data)
    density_map = 255 * density_map / np.max(density_map)
    gt_data = gt_data[0][0]
    density_map = density_map[0][0]
    if density_map.shape[1] != input_img.shape[1]:
        density_map = cv2.resize(density_map, (input_img.shape[1], input_img.shape[0]))
        gt_data = cv2.resize(gt_data, (input_img.shape[1], input_img.shape[0]))
    result_img = np.hstack((input_img, gt_data, density_map))
    cv2.imwrite(os.path.join(output_dir, fname), result_img)


def save_density_map(density_map, output_dir, fname='results.png'):
    density_map = 255 * density_map / np.max(density_map)
    density_map = density_map[0][0]
    cv2.imwrite(os.path.join(output_dir, fname), density_map)


def display_results(input_img, gt_data, density_map):
    # input_img = input_img[0][0]
    print(input_img.shape)
    gt_data = 255 * gt_data / np.max(gt_data)
    density_map = 255 * density_map / np.max(density_map)
    gt_data = gt_data[0][0]
    # density_map = density_map[0][0]
    if density_map.shape[1] != input_img.shape[1]:
        input_img = cv2.resize(input_img, (density_map.shape[1], density_map.shape[0]))
    result_img = np.hstack((input_img, gt_data, density_map))
    result_img = result_img.astype(np.uint8, copy=False)
    cv2.imshow('Result', result_img)
    cv2.waitKey(0)
dens=np.load('/home/chase/datasets/crowd_counting/ShanghaiTech/part_B_final/train_data/npy_0.25/IMG_4.npy')
show_desity_map(dens)