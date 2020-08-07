#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2020/8/7 下午7:51
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
    def preorderTraversal(self, root: TreeNode) -> list[int]:
        cur = root
        res = []
        stack = []
        while stack or cur:
            while cur:
                stack.append(cur)
                res.append(cur.val)
                cur = cur.left
            cur = stack.pop().right
        return res

        # if not root:
        #     return []
        # out = [root.val]
        # out.extend(self.preorderTraversal(root.left))
        # out.extend(self.preorderTraversal(root.right))
        # return out
