#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2020/8/5 下午11:31
# @Author  : Boyka
# @Email   : upcvagen@163.com
# @Software: PyCharm

# Definition for a binary tree node.
class TreeNode:
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None


class Solution:
    def inorderTraversal(self, root: TreeNode) -> list[int]:
        stack = []
        res = []
        cur = root
        while stack or cur:
            while cur:
                stack.append(cur)
                cur = cur.left
            cur = stack.pop()
            res.append(cur.val)
            cur = cur.right
        return res


if __name__ == '__main__':
    solution = Solution()
    print(solution.avoidFlood([1, 2, 0, 2, 3, 0, 1]))
