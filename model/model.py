#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2020/6/1 下午8:11
# @Author  : Boyka
# @Email   : upcvagen@163.com
# @File    : model.py
# @Software: PyCharm
import torch
import torch.nn as nn


class MCNNModel(nn.Module):
    def __init__(self):
        """
        3 brunches correspond to 3 different scale level filters.
        Each brunch consi
        """
        super(MCNNModel).__init__()
        self.brunch1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=9, stride=4, padding_mode='zeros'),
            nn.MaxPool2d(2),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=7, stride=3, padding_mode='zeros'),
            nn.MaxPool2d(2),
            nn.Conv2d(in_channels=32, out_channels=16, kernel_size=7, stride=3, padding_mode='zeros'),
            nn.Conv2d(in_channels=16, out_channels=8, kernel_size=7, stride=3, padding_mode='zeros')
        )

        self.brunch2 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=20, kernel_size=7, padding=3, padding_mode='zeros'),
            nn.MaxPool2d(2),
            nn.Conv2d(in_channels=20, out_channels=40, kernel_size=5, padding=2, padding_mode='zeros'),
            nn.MaxPool2d(2),
            nn.Conv2d(in_channels=40, out_channels=2, kernel_size=5, padding=2, padding_mode='zeros'),
            nn.Conv2d(in_channels=20, out_channels=10, kernel_size=5, padding=2, padding_mode='zeros')
        )

        self.brunch3 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=24, kernel_size=5, padding=2, padding_mode='zeros'),
            nn.MaxPool2d(2),
            nn.Conv2d(in_channels=24, out_channels=48, kernel_size=3, padding=1, padding_mode='zeros'),
            nn.MaxPool2d(2),
            nn.Conv2d(in_channels=48, out_channels=24, kernel_size=3, padding=1, padding_mode='zeros'),
            nn.Conv2d(in_channels=24, out_channels=12, kernel_size=3, padding=1, padding_mode='zeros')
        )
        self.fuse = nn.Sequential(nn.Conv2d(in_channels=30, out_channels=1, kernel_size=1))

    def forward(self, data):
        large_feature_map = self.brunch1(data)
        medium_feature_map = self.brunch2(data)
        small_feature_map = self.brunch3(data)
        concatenated_feature_map = torch.cat((large_feature_map, medium_feature_map, small_feature_map), 1)
        density_map = self.fuse(concatenated_feature_map)
        return density_map
