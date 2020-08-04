#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2020/8/3 下午11:46
# @Author  : Boyka
# @Email   : upcvagen@163.com
# @Software: PyCharm

class MyQueue:

    def __init__(self):
        """
        Initialize your data structure here.
        """
        self.in_stack = []
        self.out_stack = []

    def push(self, x: int) -> None:
        """
        Push element x to the back of queue.
        """
        self.in_stack.append(x)

    def pop(self) -> int:
        """
        Removes the element from in front of queue and returns that element.
        """
        if len(self.out_stack) == 0:
            while len(self.in_stack) != 0:
                self.out_stack.append(self.in_stack.pop())
        return self.out_stack.pop()

    def peek(self) -> int:

        """
        Get the front element.
        """
        if len(self.in_stack) == 0 and len(self.out_stack) == 0:
            return 0
        elif len(self.in_stack) == 0:
            return self.out_stack[-1]
        elif len(self.out_stack) == 0:
            return self.in_stack[0]

    def empty(self) -> bool:
        """
        Returns whether the queue is empty.
        """
        if len(self.in_stack) == 0 and len(self.out_stack) == 0:
            return True
        else:
            return False

# Your MyQueue object will be instantiated and called as such:
# obj = MyQueue()
# obj.push(x)
# param_2 = obj.pop()
# param_3 = obj.peek()
# param_4 = obj.empty()
