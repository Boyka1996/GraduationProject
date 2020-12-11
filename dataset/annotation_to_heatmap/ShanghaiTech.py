#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time       : 11/19/20 6:06 PM
# @Author     : Boyka
# @Contact    : upcvagen@163.com
# @File       : ShanghaiTech.py
# @Project    : GraduationProject
# @Description:
import numpy as np
import scipy
import scipy.io as io
from scipy.ndimage.filters import gaussian_filter
import scipy.spatial as spatial
import os
import glob
from matplotlib import pyplot as plt


# partly borrowed from https://github.com/davideverona/deep-crowd-counting_crowdnet
def gaussian_filter_density(img, points):
    """

    :param img:
    :param points:
    :return:
    """

    img_shape = [img.shape[0], img.shape[1]]
    print("Shape of current image: ", img_shape, ". Totally need generate ", len(points), "gaussian kernels.")
    density = np.zeros(img_shape, dtype=np.float32)
    gt_count = len(points)
    if gt_count == 0:
        return density

    leafsize = 2048
    # build kdtree
    tree = spatial.KDTree(points.copy(), leafsize=leafsize)
    # query kdtree
    distances, locations = tree.query(points, k=4)

    print('generate density...')
    for i, pt in enumerate(points):
        pt2d = np.zeros(img_shape, dtype=np.float32)
        if int(pt[1]) < img_shape[0] and int(pt[0]) < img_shape[1]:
            pt2d[int(pt[1]), int(pt[0])] = 1.
        else:
            continue
        if gt_count > 1:
            sigma = (distances[i][1] + distances[i][2] + distances[i][3]) * 0.1
        # else:
        #     sigma = np.average(np.array(gt.shape)) / 2. / 2.  # case: 1 point
        density += scipy.ndimage.filters.gaussian_filter(pt2d, sigma, mode='constant')
    print('done.')
    return density


# test code
if __name__ == "__main__":
    # show an example to use function generate_density_map_with_fixed_kernel.
    root = '/home/chase/datasets/crowd_counting/ShanghaiTech'

    # now generate the ShanghaiA's ground truth
    part_A_train = os.path.join(root, 'part_A_final/train_data', 'images')
    part_A_test = os.path.join(root, 'part_A_final/test_data', 'images')
    # part_B_train = os.path.join(root,'part_B_final/train_data','images')
    # part_B_test = os.path.join(root,'part_B_final/test_data','images')
    path_sets = [part_A_train, part_A_test]

    img_paths = []
    for path in path_sets:
        for img_path in glob.glob(os.path.join(path, '*.jpg')):
            img_paths.append(img_path)

    for img_path in img_paths:
        print(img_path)
        mat = io.loadmat(img_path.replace('.jpg', '.mat').replace('images', 'ground_truth').replace('IMG_', 'GT_IMG_'))
        img = plt.imread(img_path)  # 768行*1024列
        k = np.zeros((img.shape[0], img.shape[1]))
        points = mat["image_info"][0, 0][0, 0][0]  # 1546person*2(col,row)
        k = gaussian_filter_density(img, points)
        # plt.imshow(k,cmap=CM.jet)
        # save density_map to disk
        np.save(img_path.replace('.jpg', '.npy').replace('images', 'density_map'), k)

    '''
    #now see a sample from ShanghaiA
    plt.imshow(Image.open(img_paths[0]))

    gt_file = np.load(img_paths[0].replace('.jpg','.npy').replace('images','ground_truth'))
    plt.imshow(gt_file,cmap=CM.jet)

    print(np.sum(gt_file))# don't mind this slight variation
    '''
