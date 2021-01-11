#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time       : 2020/12/30 11:41
@Author     : Boyka
@Contact    : upcvagen@163.com
@File       : 1.py
@Project    : GraduationProject
@Description:
"""
import tensorflow as tf
from keras.utils import plot_model
# IMG_SHAPE = (224, 224, 3)
# model0 = tf.keras.applications.EfficientNetB0(input_shape=IMG_SHAPE, include_top=False, weights="imagenet")
# tf.keras.utils.plot_model(model0) # to draw and visualize
# model0.summary() # to see the list of layers and parameters
from efficientnet_pytorch import EfficientNet
model = EfficientNet.from_pretrained('efficientnet-b7')