#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2020/8/25 下午8:03
# @Author  : Boyka
# @Email   : upcvagen@163.com
# @Software: PyCharm

class Solution:
    def commonChars(self, chars):
        a = [0 for _ in range(26)]
        b = ""
        for j in chars[0]:
            a[ord(j) - ord('a')] += 1
        for i in range(len(a)):
            if a[i] == 0:
                continue
            for j in range(1, len(chars)):
                a[i] = min(a[i], chars[j].count(chr(i + ord('a'))))
        for i, ele in enumerate(a):
            if ele == 0:
                continue
            for x in range(ele):
                b += chr(i + ord('a'))
        return b
        # res=list()
        # for w in set(chars[0]):
        #     count=[x.count(w) for x in chars]
        #     a=w*min(count)
        #     for i in a:
        #         res+=i
        # return res
        # res = []
        # if not chars:
        #     return res
        # key = set(chars)
        # for k in key:
        #     minimum = min(a.count(k) for a in chars)
        #     res += minimum * k
        # return res


if __name__ == '__main__':
    solution = Solution()
    a = ["bella", "label", "roller"]
    print(solution.commonChars(a))
