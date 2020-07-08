#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2020/7/6 下午11:45
# @Author  : Boyka
# @Email   : upcvagen@163.com
# @Software: PyCharm
import json
if __name__ == '__main__':

    json_path='/home/chase/datasets/crowd_counting/ShanghaiTech/part_B_final/test_data/json/IMG_1.json'
    with open(json_path,'r') as fr:
        data = json.load(fr)
    print(len(data.get('points')))