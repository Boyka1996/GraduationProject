#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2020/4/12 上午10:31
# @Author  : Boyka
# @Email   : upcvagen@163.com
# @File    : producer_customer.py
# @Software: PyCharm
__author__ = "JentZhang"

import time, threading, queue
import cv2



def test(name):
    q = queue.Queue(maxsize=3)  # 声明队列

    def Producer(video_path):
        cap = cv2.VideoCapture(video_path)
        if cap.isOpened:
            while True:
                try:
                    ret, frame = cap.read()
                    # logger.info(ret)
                    if not ret:
                        break
                    q.put(frame)
                    time.sleep(0.02)
                except:
                    pass


    def Consumer(video):
        '''消费者'''

        while True:
            i = q.get()  # 从队列中取数据
            print(video)
            print(i.shape)
            time.sleep(0.5)

    '''设置多线程'''
    p = threading.Thread(target=Producer, args=("/home/chase/projects/00013.MTS",))
    print(p)
    c1 = threading.Thread(target=Consumer, args=(name))
    print(c1)
    # c2 = threading.Thread(target=Consumer, args=("李四",))

    '''线程开启'''
    p.start()
    c1.start()
    # c2.start()
def main():
    main_thread = threading.Thread(test("1"))
    main_thread.start()
    main_thread1 = threading.Thread(test("2"))
    main_thread1.start()
if __name__ == '__main__':
    m= threading.Thread(main())
    print("***")
    print(m)
    m.start()