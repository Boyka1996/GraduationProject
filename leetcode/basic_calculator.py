#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2020/4/19 下午10:52
# @Author  : Boyka
# @Email   : upcvagen@163.com
# @File    : basic_calculator.py
# @Software: PyCharm
class Solution(object):
    def calculate(self, s):
        """
        :type s: str
        :rtype: int
        """
        operator = []
        number = []
        for ele_id, element in enumerate(self.str2int(s)):
            if element == "(":
                operator.append(element)
            elif isinstance(element, int):
                number.append(element)
            elif element == "+" or element == "-":
                if not operator or (operator[-1] != "+" and operator[-1] != "-"):
                    operator.append(element)
                elif operator[-1] == "+" or operator[-1] == "-":
                    number.append(operator.pop())
                    operator.append(element)

            elif element == ")":
                temp = operator.pop()
                while True:
                    if temp != "(":
                        number.append(temp)
                    else:
                        break
                    temp = operator.pop()
        while operator:
            number.append(operator.pop())
        # print(operator)
        print(number)
        result = []
        for num in number:
            if isinstance(num, int):
                result.append(num)
                # print()
            elif num == "+":
                temp = result.pop()
                result.append(temp + result.pop())
            elif num == "-":
                temp = result.pop()
                result.append(result.pop() - temp)
                # result.append(int(temp) - int(result.pop()))
        # print(result)
        return result[0]

    @staticmethod
    def str2int(s):
        result_list = []
        num = 0
        tag = False
        b=[]
        for element in list(s):
            if element!=' ':
                b.append(element)
        # print(b)
        for ele_id, element in enumerate(b):
            if ele_id == len(list(b))-1:
                if element.isdigit():
                    num = num * 10 + int(element)
                    result_list.append(num)
                    break
                else:
                    if tag:
                        result_list.append(num)
                    result_list.append(element)
                    break
            if not element.isdigit():
                if tag:
                    result_list.append(num)
                    num = 0
                    tag = False
                result_list.append(element)
            else:
                tag = True
                num = num * 10 + int(element)
                if ele_id == len(list(b)) - 1:
                    result_list.append(num)
        # print(result_list)
        return result_list


if __name__ == '__main__':
    solution = Solution()
    string = " 2-1 + 2 "
    string = "(1+(4+5+2)-3)+(6+8)"
    string = "2147483647"
    string="(1-(3-4))"

    print(len(string))
    result_num = solution.calculate(string)
    print(result_num)
    print(type(result_num))
