#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2020/6/15 上午8:15
# @Author  : Boyka
# @Email   : upcvagen@163.com
# @Software: PyCharm

class Solution(object):
    def reverse(self, x):
        """
        :type x: int
        :rtype: int
        """
        if x > 2 ** 31 - 1 or x < -2 ** 31:
            return 0
        tag = 1 if x > 0 else -1
        x *= tag
        result = 0
        while True:
            remainder = x % 10
            result = result * 10 + remainder
            if x // 10 == 0:
                break
            x = x // 10
        result = result * tag
        return result if result < 2 ** 31 - 1 and result > -2 ** 31 else 0


if __name__ == '__main__':
    solution = Solution()
    print(solution.reverse(1534236469))
