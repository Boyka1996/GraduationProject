#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2020/6/13 上午11:49
# @Author  : Boyka
# @Email   : upcvagen@163.com
# @Software: PyCharm

class Solution(object):
    def isPalindrome(self):
        """
        :type x: int
        :rtype: bool
        """
        s = str(12321)
        print(len(s))
        print(len(s) / 2)
        # half=
        for i in range(int(len(s) / 2)):
            print(i)
            print(s[i], s[-i-1])
            if s[i] != s[-i-1]:
                return False
        return True


if __name__ == '__main__':
    solution = Solution()
    print(solution.isPalindrome())
