#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2020/7/25 下午10:18
# @Author  : Boyka
# @Email   : upcvagen@163.com
# @Software: PyCharm
class Solution:
    def maxArea(self, height) -> int:
        length = len(height)
        if length < 2:
            return 0
        most_water = 0
        left = 0
        right=length-1
        while left < right:
            most_water = max((right - left) * min(height[right], height[left]), most_water)
            if height[left] > height[right]:
                right -= 1
            else:
                left += 1
        return most_water


if __name__ == '__main__':
    solution = Solution()
    print(solution.maxArea([1, 8, 6, 2, 5, 4, 8, 3, 7]))
