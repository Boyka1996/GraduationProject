#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2020/7/23 下午2:55
# @Author  : Boyka
# @Email   : upcvagen@163.com
# @Software: PyCharm

# class Solution:
#     def myAtoi(self, str: str) -> int:
#         print("9"-"0")
#         if not str or str == "":
#             return 0
#         result = 0
#         INT_MAX = pow(2, 32)
#         INT_MIN = pow(-2, 31)
#         if_started = False
#         if_positive = 1
#         if_first_num = False
#         ifstart = False
#         for c in str:
#             if c == " ":
#                 continue
#             if not if_started:
#                 if not c.isdigit() and not c == "-" and not c == "+":
#                     break
#                 elif c == "+":
#                     if_started = True
#                     ifstart = True
#                 elif c == "-":
#                     if_positive = -1
#                     if_started = True
#                     ifstart = True
#                 else:
#                     result = result * 10 + int(c)
#                     if_started = True
#                     ifstart = True
#                     if c == "0":
#                         if_first_num = True
#                     else:
#                         if_first_num = False
#             else:
#                 if ifstart and c == "0" and if_first_num:
#                     continue
#                 elif c.isdigit():
#                     ifstart = False
#                     result = result * 10 + int(c)
#                 else:
#                     break
#
#         return max(INT_MIN, min(INT_MAX, result * if_positive))

class Solution:
    def myAtoi(self, str: str) -> int:
        if not str or str == "":
            return 0
        result = 0
        INT_MAX = pow(2, 31) - 1
        INT_MIN = pow(-2, 31)
        if_started = False
        if_positive = 1

        for c in str:
            if c == " " and not if_started:
                continue
            elif not if_started:
                if not c.isdigit() and not c == "-" and not c == "+":
                    break
                elif c == "+":
                    if_started = True
                elif c == "-":
                    if_positive = -1
                    if_started = True
                else:
                    result = result * 10 + int(c)
            else:
                if c.isdigit():
                    result = result * 10 + int(c)
                else:
                    break

        return max(INT_MIN, min(INT_MAX, result * if_positive))


if __name__ == '__main__':
    solution = Solution()
    print(solution.myAtoi("   +0 123"))
