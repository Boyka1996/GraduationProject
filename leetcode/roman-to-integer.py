#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2020/7/29 上午9:33
# @Author  : Boyka
# @Email   : upcvagen@163.com
# @Software: PyCharm

class Solution:
    def romanToInt(self, s: str) -> int:
        res = 0
        letter_dict = {"M": 1000, "D": 500, "C": 100, "L": 50, "X": 10, "V": 5, "I": 1}
        for c_id in range(len(s)):
            if c_id < len(s) - 1 and letter_dict[s[c_id]] < letter_dict[s[c_id + 1]]:
                res -= letter_dict[s[c_id]]
            else:
                res += letter_dict[s[c_id]]
        return res


if __name__ == '__main__':
    solution = Solution()
    print(solution.romanToInt("MCMXCIV"))
