#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2020/7/23 上午10:03
# @Author  : Boyka
# @Email   : upcvagen@163.com
# @Software: PyCharm

class Solution:
    def longestPalindrome(self, s: str) -> str:
        if not s:
            return s
        start = 0
        max_len = 0
        dp = [[False for i in range(len(s))] for j in range(len(s))]
        for i in range(len(s)):
            dp[i][i] = True
        for i in range(1, len(s)):
            for j in range(i):
                if s[i] != s[j]:
                    continue
                else:
                    if dp[i - 1][j + 1] or i - j < 3:
                        dp[i][j] = True
                        current_len = i - j + 1
                        if current_len > max_len:
                            start = j
                            max_len = current_len
                    else:
                        continue
        if max_len == 0:
            return s[0]

        return s[start:start + max_len]


if __name__ == '__main__':
    solution = Solution()
    print(solution.longestPalindrome(""))
