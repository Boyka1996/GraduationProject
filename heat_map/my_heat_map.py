#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2020/5/26 上午11:24
# @Author  : Boyka
# @Email   : upcvagen@163.com
# @File    : my_heat_map.py
# @Software: PyCharm

from math import log, sqrt

import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import multivariate_normal


def cal_sigma(dmax, edge_value):
    return sqrt(- pow(dmax, 2) / log(edge_value))


def gaussian(array_like_hm, mean, sigma):
    """modifyed version normal distribution pdf, vector version"""
    array_like_hm -= mean
    x_term = array_like_hm[:, 0] ** 2
    y_term = array_like_hm[:, 1] ** 2
    exp_value = - (x_term + y_term) / 2 / pow(sigma, 2)
    return np.exp(exp_value)


def draw_heatmap(width, height, x, y, sigma, array_like_hm):
    m1 = (x, y)
    s1 = np.eye(2) * pow(sigma, 2)
    # k1 = multivariate_normal(mean=m1, cov=593.109206084)
    k1 = multivariate_normal(mean=m1, cov=s1)
    #     zz = k1.pdf(array_like_hm)
    zz = gaussian(array_like_hm, m1, sigma)
    img = zz.reshape((height, width))
    return img


def test(width, height, x, y, array_like_hm):
    dmax = 100
    edge_value = 0.01
    sigma = cal_sigma(dmax, edge_value)

    return draw_heatmap(width, height, x, y, sigma, array_like_hm)


xres = 1920
yres = 1080
xlim = (0, xres)
ylim = (0, yres)

# x = np.linspace(xlim[0], xlim[1], xres)
# y = np.linspace(ylim[0], ylim[1], yres)
x = np.arange(xres, dtype=np.float)
y = np.arange(yres, dtype=np.float)
xx, yy = np.meshgrid(x, y)

# evaluate kernels at grid points
xxyy = np.c_[xx.ravel(), yy.ravel()]
# %timeit img = test(1024, 576, 512, 288, xxyy.copy())
img = test(xres, yres, xres / 2, yres / 2, xxyy.copy())
plt.imshow(img)
plt.show()
