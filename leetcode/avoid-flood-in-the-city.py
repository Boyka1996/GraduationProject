#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2020/7/26 下午9:50
# @Author  : Boyka
# @Email   : upcvagen@163.com
# @Software: PyCharm

class Solution:
    def avoidFlood(self, rains):
        result = [1 for _ in range(len(rains))]
        sun_list = []
        rain_dict = dict()

        for day, rain in enumerate(rains):
            if rain > 0:
                result[day] = -1
                if not rain in rain_dict:
                    rain_dict[rain] = [day]
                else:
                    rain_dict[rain] = [rain_dict[rain][0], day]
            elif rain == 0:
                sun_list.append(day)
            else:
                return []
        for lake in rain_dict:
            if len(rain_dict[lake]) == 1:
                continue
            else:
                for start_id in range(int(len(rain_dict[lake]) / 2)):
                    start = rain_dict[lake][2 * start_id]
                    end = rain_dict[lake][2 * start_id + 1]
                    if_drain = False
                    for sun in sun_list:
                        if start < sun < end:
                            result[sun] = lake
                            if_drain = True
                            sun_list.remove(sun)
                            break
                    if not if_drain:
                        return []
        return result


if __name__ == '__main__':
    solution = Solution()
    print(solution.avoidFlood([69,0,0,0,69]))
