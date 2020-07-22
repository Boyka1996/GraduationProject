#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2020/6/12 下午3:05
# @Author  : Boyka
# @Email   : upcvagen@163.com
# @File    : longest-substring-without-repeating-characters.py
# @Software: PyCharm

"""
3. 无重复字符的最长子串

30. 串联所有单词的子串

76. 最小覆盖子串

159. 至多包含两个不同字符的最长子串

209. 长度最小的子数组

239. 滑动窗口最大值

567. 字符串的排列

632. 最小区间

727. 最小窗口子序列

作者：powcai
链接：https://leetcode-cn.com/problems/longest-substring-without-repeating-characters/solution/hua-dong-chuang-kou-by-powcai/
来源：力扣（LeetCode）
著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。
"""
class Solution(object):

    def lengthOfLongestSubstring(self, s):
        """
        :type s: str
        :rtype: int
        """
        if not s:
            return 0
        left = 0
        max_size = 0
        sub_string = set()
        for i in range(len(s)):
            while s[i] in sub_string:
                sub_string.remove(s[left])
                left += 1

            sub_string.add(s[i])
            max_size = max(max_size, len(sub_string))
        return max_size



if __name__ == '__main__':
    solution = Solution()
    print(solution.lengthOfLongestSubstring("pwwkew"))
