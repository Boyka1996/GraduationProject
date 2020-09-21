#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2020/8/25 下午7:50
# @Author  : Boyka
# @Email   : upcvagen@163.com
# @Software: PyCharm
#
#
# @param str1 string字符串 原始的字符串
# @param str2 string字符串 转换的字符串
# @return string字符串
#


class Solution:
    def find_diff_char(self, str1, str2):
        n = 0
        for c in str1:
            n ^= ord(c)
        for c in str2:
            n ^= ord(c)
        return chr(n)

        # if not str1: return str2
        # if not str2: return str1
        # for key, val in collections.Counter(str2).items():
        #     if not str1.count(key) == val:
        #         return key


if __name__ == '__main__':
    solution = Solution()
    print(solution.find_diff_char("abcd", "abcde"))
