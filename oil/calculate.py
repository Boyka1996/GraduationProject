#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time       : 2021/1/19 22:45
@Author     : Zhougou
@Contact    : upcvagen@163.com
@File       : calculate.py
@Project    : GraduationProject
@Description:
"""
import numpy as np
import pandas as pd

df = pd.read_table('A3.csv', header=None, sep=',', engine='python')

df_layer = pd.read_table('A3分层.csv', header=None, sep=',',engine='python')

row_df_layer = df_layer.shape[0]  # 行数
col_df_layer = df_layer.shape[1]  # 列数
df_layer_num = df_layer.iloc[1:row_df_layer, [0]]  # 分层数据的层号
df_layer_start = df_layer.iloc[1:row_df_layer, [1]]  # 分层数据的开始位置
df_layer_end = df_layer.iloc[1:row_df_layer:, [2]]  # 分层数据的终止位置

row_df = df.shape[0]
col_df = df.shape[1]
DEPTH = df.iloc[1:row_df, [0]]
AC = df.iloc[1:row_df, [1]]
SP = df.iloc[1:row_df, [2]]
GR = df.iloc[1:row_df, [3]]
RT = df.iloc[1:row_df, [4]]

DEPTH_f = DEPTH.values.astype(np.float)  # str转float
AC_f = AC.values.astype(np.float)
SP_f = SP.values.astype(np.float)
GR_f = GR.values.astype(np.float)
RT_f = AC.values.astype(np.float)

# print(type(df_layer_start))
end_f = df_layer_end.values.astype(np.float)  # str转float
start_f = df_layer_start.values.astype(np.float)  # str转float
H = end_f - start_f
i = 1
j = 0
temp = 0
print(row_df_layer)
print(DEPTH_f)
print(start_f[j], DEPTH_f[i], end_f[j])
# print(len(row_df_layer))
for j in range(len(df_layer_start)):
    # print(j)
    # print(start_f[j])
    if(start_f[j]<DEPTH_f[i])&(DEPTH_f[i]<end_f[j]):
        a=SP_f[i]*i
        temp=a+temp
        print(temp)