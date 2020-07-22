#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2020/7/22 上午10:52
# @Author  : Boyka
# @Email   : upcvagen@163.com
# @Software: PyCharm
class Solution(object):

    def minSubArrayLen(self, s, nums):
        if not s or not nums or sum(nums) < s:
            return 0
        left = 0
        num_sum = 0
        start = 0
        end = len(nums)
        for i in range(len(nums)):
            num_sum += nums[i]
            if num_sum >= s:
                while sum(nums[left:i + 1]) >= s:
                    num_sum -= nums[left]
                    left += 1
                if i + 1 - left < end - start:
                    start = left - 1
                    end = i
        return end - start + 1


if __name__ == '__main__':
    solution = Solution()
    # print(solution.minSubArrayLen(s=100, nums=[]))
    print(solution.minSubArrayLen(s=7, nums=[2, 3, 1, 2, 4, 3]))
