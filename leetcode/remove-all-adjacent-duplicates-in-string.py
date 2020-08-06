#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2020/8/5 下午11:13
# @Author  : Boyka
# @Email   : upcvagen@163.com
# @Software: PyCharm
class Solution:
    def removeDuplicates(self, S):
        if len(S) < 2:
            return S
        stack = []
        for char in S:
            if stack and stack[-1] == char:
                stack.pop()
            else:
                stack.append(char)

        return ''.join(stack)


if __name__ == '__main__':
    solution = Solution()

    print(solution.removeDuplicates("abbaca"))
