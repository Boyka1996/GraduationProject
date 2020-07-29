#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2020/7/29 下午3:39
# @Author  : Boyka
# @Email   : upcvagen@163.com
# @Software: PyCharm

class Solution:
    def removeOuterParentheses(self, S: str) -> str:
        my_list = []
        res = ""
        for c in S:
            if c == "(":
                my_list.append("(")
                if len(my_list) > 1:
                    res += "("
            elif c == ")":
                my_list.pop()
                if len(my_list) > 0:
                    res += ")"

        return res


if __name__ == '__main__':
    solution = Solution()
    print(solution.removeOuterParentheses("(()())(())"))
