#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2020/7/31 下午10:24
# @Author  : Boyka
# @Email   : upcvagen@163.com
# @Software: PyCharm



class CQueue:
    def __init__(self):
        self.in_stack = []
        self.out_stack = []

    def appendTail(self, value: int) -> None:
        # while len(self.out_stack) != 0:
        #     self.in_stack.append(self.out_stack.pop())
        self.in_stack.append(value)

    def deleteHead(self) -> int:
        if len(self.out_stack) != 0:
            return self.out_stack.pop()
        elif len(self.in_stack) != 0:
            while len(self.in_stack) != 0:
                self.out_stack.append(self.in_stack.pop())
            return self.out_stack.pop()
        else:
            return -1

    # ["CQueue",
    # "deleteHead", "appendTail", "deleteHead", "appendTail", "appendTail",
    # "deleteHead", "deleteHead", "deleteHead", "appendTail", "deleteHead",
    # "appendTail", "appendTail", "appendTail", "appendTail", "appendTail",
    # "appendTail", "deleteHead", "deleteHead", "deleteHead", "deleteHead"]
    # [[],
    # [], [12], [], [10], [9],
    # [], [], [], [20], [],
    # [1], [8], [20], [1], [11],
    # [2], [], [], [], []]
    # def __init__(self):
    #     self.stack1 = []
    #     self.stack2 = []
    #
    # def appendTail(self, value: int) -> None:
    #     if not self.stack1:
    #         self.stack1.append(value)
    #     else:
    #         self.stack2.append(value)
    #
    #
    # def deleteHead(self) -> int:
    #     if len(self.stack1) == 0 and len(self.stack2) == 0:
    #         return -1
    #     elif len(self.stack1) != 0:
    #         while len(self.stack1) > 1:
    #             self.stack2.append(self.stack1.pop())
    #         return self.stack1.pop()
    #     elif len(self.stack2) != 0:
    #         while len(self.stack2) > 1:
    #             self.stack1.append(self.stack2.pop())
    #         return self.stack2.pop()
    # if not self.stack1 and not self.stack2:
    #     return -1
    # elif not self.stack1:
    #     while len(self.stack1)>1:
    #         self.stack2.append(self.stack1.pop())
    #     return self.stack1.pop()
    # elif not self.stack2:
    #     while len(self.stack2)>1:
    #         self.stack1.append(self.stack2.pop())
    #     return self.stack2.pop()

# Your CQueue object will be instantiated and called as such:
# obj = CQueue()
# obj.appendTail(value)
# param_2 = obj.deleteHead()
