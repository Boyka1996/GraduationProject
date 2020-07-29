#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2020/7/27 下午11:28
# @Author  : Boyka
# @Email   : upcvagen@163.com
# @Software: PyCharm

# class Solution:
#     def intToRoman(self, num: int) -> str:
#         num_list = [["", "M", "MM", "MMM"],
#                     ["", "C", "CC", "CCC", "CD", "D", "DC", "DCC", "DCCC", "CM"],
#                     ["", "X", "XX", "XXX", "XL", "L", "LX", "LXX", "LXXX", "XC"],
#                     ["", "I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX"]]
#         res = ""
#         try:
#             res+=num_list[0][int(num / 1000)]
#             res += num_list[1][int(num % 1000 / 100)]
#             res += num_list[2][int(num % 100 / 10)]
#             res += num_list[3][int(num % 10)]
#         finally:
#             return res
# 贪心
class Solution:
    def intToRoman(self, num: int) -> str:
        res = ""
        nums = [1000, 900, 500, 400, 100, 90, 50, 40, 10, 9, 5, 4, 1]
        romans = ["M", "CM", "D", "CD", "C", "XC", "L", "XL", "X", "IX", "V", "IV", "I"]

        for n_id, n in enumerate(nums):
            while num >=n:
                res += romans[n_id]
                num -= n
        return res


if __name__ == '__main__':
    solution = Solution()
    print(solution.intToRoman(1994))
