#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2020/7/29 下午4:40
# @Author  : Boyka
# @Email   : upcvagen@163.com
# @Software: PyCharm

# Definition for a binary tree node.
class TreeNode:
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None


class BSTIterator:

    def __init__(self, root: TreeNode):
        self.stack = []
        while root:
            self.stack.append(root)
            root = root.left

    def next(self) -> int:
        """
        @return the next smallest number
        """
        tmp = self.stack.pop()
        res = tmp.val
        tmp=tmp.right
        while tmp:
            self.stack.append(tmp)
            tmp = tmp.left
        return res

    def hasNext(self) -> bool:
        """
        @return whether we have a next smallest number
        """
        return self.stack != []

# Your BSTIterator object will be instantiated and called as such:
# obj = BSTIterator(root)
# param_1 = obj.next()
# param_2 = obj.hasNext()
