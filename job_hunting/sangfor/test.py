#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2020/8/25 下午7:47
# @Author  : Boyka
# @Email   : upcvagen@163.com
# @Software: PyCharm

#coding=utf-8
# 本题为考试单行多行输入输出规范示例，无需提交，不计分。
import sys
for line in sys.stdin:
    a = line.split()
    print(int(a[0]) + int(a[1]))