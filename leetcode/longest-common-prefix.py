#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2020/6/15 上午9:51
# @Author  : Boyka
# @Email   : upcvagen@163.com
# @Software: PyCharm

class Solution(object):
    def longestCommonPrefix(self, strs):
        """
        :type strs: List[str]
        :rtype: str
        """
        result = ""
        for i in zip(*strs):
            if len(set(i)) == 1:
                result += i[0]
            else:
                break
        return result


if __name__ == '__main__':
    solution = Solution()
    print(solution.longestCommonPrefix(["abc", "abcd", "azcd", "aacc"]))
