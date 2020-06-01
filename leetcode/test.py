#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2020/4/20 上午12:35
# @Author  : Boyka
# @Email   : upcvagen@163.com
# @File    :
# @Software: PyCharm

def str2int(s):
    result_list = []
    num = 0
    tag = False
    for ele_id,element in enumerate(list(s)):
        if ele_id==len(list(s))-1 and tag:
            result_list.append(num)
            break
        if not element.isdigit():
            if tag:
                result_list.append(num)
                num=0
                tag=False
            result_list.append(element)
        else:
            tag = True
            num = num * 10 + int(element)
    return result_list
if __name__ == '__main__':
    s="(1-(3-4))"
    s="1000"
    print(str2int(s))
