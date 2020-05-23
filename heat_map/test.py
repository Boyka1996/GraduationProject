#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2020/5/22 下午3:09
# @Author  : Boyka
# @Email   : upcvagen@163.com
# @File    : test.py
# @Software: PyCharm
import numpy as np

import numpy

a = numpy.array([
    [[5, 10, 15],
     [10, 20, 30],
     [20, 40, 60]],

    [[5, 10, 15],
     [10, 20, 30],
     [20, 40, 60]]
])
# 按行相加，此时第一、二行都是[5,10,15],[10,20,30],[20,40,60]
b = a.sum(axis=1)
# 按列相加，此时第一列是[5,10,15],第二列是[10,20,30]，第三列是[20,40,60]
c = a.sum(axis=0)
d=a.sum()
print(b)
print(c)
print(d)
