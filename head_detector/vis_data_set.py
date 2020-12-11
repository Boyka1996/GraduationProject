import cv2
import os
import xml.etree.ElementTree as ET


def get_points(file_path):
    tree = ET.ElementTree(file=file_path)
    root = tree.getroot()
    points = root.findall("object")
    obj_points = []
    for point in points:
        obj = []
        bndbox = point.find("bndbox")
        obj.append(point.find("name").text)
        obj.append(int(bndbox.find("xmin").text))
        obj.append(int(bndbox.find("ymin").text))
        obj.append(int(bndbox.find("xmax").text))
        obj.append(int(bndbox.find("ymax").text))
        obj_points.append(obj)
    return obj_points


def vis_image(image, points):
    print(points)
    RED = (0, 0, 255)
    GREEN = (0, 255, 0)
    for point in points:
        font = cv2.FONT_HERSHEY_COMPLEX
        cv2.putText(image, point[0], (point[1], point[2]), font, color=GREEN, thickness=1, fontScale=0.6)
        cv2.rectangle(img=image, pt1=(point[1], point[2]), pt2=(point[3], point[4]), thickness=1, color=RED)
    cv2.imshow('1', image)
    cv2.waitKey(0)


if __name__ == '__main__':
    img_path = "/home/chase/datasets/crowd_counting/SCUT/SCUT_HEAD_Part_A/JPEGImages/PartA_00000.jpg"
    xml_path = "/home/chase/datasets/crowd_counting/SCUT/SCUT_HEAD_Part_A/Annotations/PartA_00000.xml"
    vis_image(cv2.imread(img_path), get_points(xml_path))
