# !/usr/bin/python3
# -*- codeing = utf-8 -*-
# @Time : 4/25/2021 9:35 PM
# @Author : Liu
# @File : PDSP_FOR.py
# @Software : PyCharm
import numpy as np
import pandas as pd


def PSNP(train_data, test_data, train_label):
    # 获取DNA长度
    DNA_len = train_data[0].__len__()    # 默认train和test的dna大小相同

    # 获取行数，也就是DNA的数据个数
    train_row_number = train_data.__len__()
    test_row_number = test_data.__len__()

    # 统计0、1的个数
    counst_0_1 =pd.DataFrame(train_label).value_counts()
    # one_dna
    one_dna= ['A', 'U', 'C', 'G']

    # 定义存储每个tow_dna在基因段中按位置出现的个数总数:A
    num_true = np.zeros((4, DNA_len), dtype=float)
    num_false = np.zeros((4, DNA_len), dtype=float)

    for i in range(0, DNA_len):
        for j in range(0, 4):
            for k in range(0,train_row_number):
                if train_label[k] == 1:
                    if train_data[k][i] == one_dna[j]:
                        num_true[j,i] = num_true[j,i]+1
                if train_label[k] == 0:
                    if train_data[k][i] == one_dna[j] :
                        num_false[j,i] = num_false[j,i]+1

    freq= num_true/counst_0_1[1] - num_false/counst_0_1[0]

    # 定义返回值
    train_psnp = np.zeros((train_row_number, DNA_len), dtype=float)
    test_psnp = np.zeros((test_row_number, DNA_len), dtype=float)

    # 分别对train,test数据进行赋值freq
    for i in range(0, DNA_len):
        for j in range(0, 4):
            for k in range(0,train_row_number):
                if train_data[k][i] == one_dna[j]:
                    train_psnp[k,i] = freq[j,i]
            for k in range(0,test_row_number):
                if test_data[k][i] == one_dna[j]:
                    test_psnp[k,i] = freq[j,i]

    return train_psnp, test_psnp



###### 一些测试代码 #######



