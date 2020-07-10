#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2020/7/10 下午3:06
# @Author  : Boyka
# @Email   : upcvagen@163.com
# @Software: PyCharm
import os
import xml.etree.ElementTree as ET

import cv2


def __indent(elem, level=0):
    i = "\n" + level * "\t"
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = i + "\t"
        if not elem.tail or not elem.tail.strip():
            elem.tail = i
        for elem in elem:
            __indent(elem, level + 1)
        if not elem.tail or not elem.tail.strip():
            elem.tail = i
    else:
        if level and (not elem.tail or not elem.tail.strip()):
            elem.tail = i


def get_annotation(xml_path):
    xml_tree = ET.parse(xml_path)
    objects = xml_tree.getroot().findall('object')
    # print(root.text)
    object_list = dict()
    for obj in objects:
        obj_dict = dict()
        obj_dict['name'] = obj.find('name').text
        bbox = obj.find('bndbox')
        for coor in bbox:
            obj_dict[coor.tag] = int(coor.text)
        object_list[obj] = obj_dict
        # object_list.append(object_list)
        print(obj_dict)

    return object_list


def draw_img(cv_img, points):
    for point in points.values():
        print(point['xmin'])
        cv2.rectangle(cv_img, (int(point['xmin'] * 640 / 4096), int(point['ymin'] * 480 / 2304)),
                      (int(point['xmax'] * 640 / 4096), int(point['ymax'] * 480 / 2304)), (0, 0, 255), 2)
    cv2.imshow('1', cv_img)
    cv2.waitKey()
    return cv_img


if __name__ == '__main__':
    xml_path = '/home/chase/Downloads/死鸡数据/死鸡数据/Annotations/IMG20191029145337.xml'
    img_folder = '/home/chase/Downloads/死鸡数据/死鸡数据/image_2'
    img_path = os.path.join(img_folder, xml_path.split('/')[-1].replace('.xml', '.jpg'))
    points = get_annotation(xml_path)
    img = cv2.imread(img_path)
    draw_img(img, points)
