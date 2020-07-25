#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2020/7/24 下午11:14
# @Author  : Boyka
# @Email   : upcvagen@163.com
# @Software: PyCharm

class Solution:
    def convert(self, s: str, numRows: int) -> str:
        result = ""
        length = len(s)
        if numRows <= 0:
            return result
        elif numRows == 1 or length <= numRows:
            return s
        elif numRows == 2:
            result = s[0::2] + s[1::2]
            return result
        else:
            group = 2 * numRows - 2
            for i in range(numRows):
                if i == 0 or i == numRows - 1:
                    result += s[i::group]
                    # tag = i
                    # while tag <= length - 1:
                    #     result += s[tag]
                    #     tag += group

                else:

                    # tag = 2 * numRows - i - 2
                    # result += s[i,tag::group]
                    # while tag <= length - 1:
                    #     result += s[tag]
                    #     tag += group

                    tag1 = i
                    tag2 = 2 * numRows - i - 2
                    while tag1 <= length - 1 or tag2 <= length - 1:
                        if tag1 <= length - 1:
                            result += s[tag1]
                            tag1 += group
                        if tag2 <= length - 1:
                            result += s[tag2]
                            tag2 += group
            # print(result)
            return result
# class Solution:
#     def convert(self, s: str, numRows: int) -> str:
#         if numRows < 2: return s
#         res = ["" for _ in range(numRows)]
#         i, flag = 0, -1
#         for c in s:
#             res[i] += c
#             if i == 0 or i == numRows - 1: flag = -flag
#             i += flag
#         return "".join(res)


if __name__ == '__main__':
    solution = Solution()
    print("LDREOEIIECIHNTSG" == solution.convert("LEETCODEISHIRING", 4))
