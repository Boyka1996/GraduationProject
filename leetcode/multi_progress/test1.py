#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2020/4/9 下午3:01
# @Author  : Boyka
# @Email   : upcvagen@163.com
# @File    : test1.py
# @Software: PyCharm
import os
from multiprocessing import Process


def info(title):
    print(title)
    print('module name:', __name__)
    print('parent process:', os.getppid())
    print('process id:', os.getpid())


def f(name):
    info('function f')
    print('hello', name)


if __name__ == '__main__':
    info('main line')
    p = Process(target=f, args=('bob',))
    p.start()
    p.join()
