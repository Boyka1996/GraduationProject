#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2020/7/10 上午11:05
# @Author  : Boyka
# @Email   : upcvagen@163.com
# @Software: PyCharm
import  sys
def solution():
    # filters = []
    scale = 1
    length = sys.stdin.readline()
    for i in range(int(length)):
        my_filter = sys.stdin.readline().split(' ')
        my_filter= list(map(int, my_filter))
        if my_filter[0] == 1:
            scale += my_filter[1]
        elif my_filter[0] == 2:
            scale *= my_filter[1]
        elif my_filter[0] == 3:
            if my_filter[1] == 0:
                return 0
            else:
                scale /= my_filter[1]
                scale=int(scale)
    # return int(scale)
    sys.stdout.write(str(scale))


if __name__ == '__main__':
    filters_demo = [[1, 6], [2, 3], [3, 3], [2, 3], [2, 3], [3, 7]]
    length_demo = len(filters_demo)
    print(solution())
