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
    one_dna = ['A', 'U', 'C', 'G']

    # 定义存储每个tow_dna在基因段中按位置出现的个数总数  标签A
    num_true = np.zeros((4, DNA_len), dtype=float)
    num_false = np.zeros((4, DNA_len), dtype=float)

    # 将数据按1与0分开
    train_data_1_str = pd.Series(train_data)[train_label == 1].str
    train_data_0_str = pd.Series(train_data)[train_label == 0].str

    for i in range(0, DNA_len):
        for j in range(0, 4):
            try:
                num_true[j, i] = (train_data_1_str[i]==one_dna[j]).value_counts()[True]     # 分类累加，然后把true选择出来
            except KeyError as e:
                num_true[j, i] = 0  # 没有True时，赋值0
            try:
                num_false[j, i] = (train_data_0_str[i]==one_dna[j]).value_counts()[True]    # 分类累加，然后把true选择出来
            except KeyError as e:
                num_false[j,i] = 0  # 没有True时，赋值0

    freq = num_true/counst_0_1[1] - num_false/counst_0_1[0]

    # 定义返回值
    train_psnp = np.zeros((train_row_number, DNA_len), dtype=float)
    test_psnp = np.zeros((test_row_number, DNA_len), dtype=float)

    # 构建str
    train_str = pd.Series(train_data).str
    test_str = pd.Series(test_data).str
    # 分别对train,test数据进行赋值freq
    for i in range(0, DNA_len):
        for j in range(0, 4):
            train_psnp[train_str[i] == one_dna[j], i] = freq[j, i]
            test_psnp[test_str[i] == one_dna[j], i] = freq[j, i]

    return train_psnp, test_psnp


###### 一些测试代码 #######
# myfind.groupby(myfind.index).sum(); 按重复索引累加
# test = pd.Series(['dacnddcc' , 'dac', 'aac'])
# a = test.str.len()
# c = a.iloc[0]
# d = test.size
# macth_str = '(' +test.str[1] + '.{'+ str(2) +'}' + test.str[2]
# test.str.extractall('(d.{1}c)')
#
# series = pd.Series(['ASDFQWEFASDF','sdf', 'QEGGRGQFSAFAD'])
# regexp = re.compile(r"(A)")
# matches = series.apply(lambda r: list(regexp.finditer(r)))
# a = matches.explode().apply(lambda r: pd.Series({"start": r.span()[0],  "match": r.group(0) }) if isinstance(r,re.Match) else pd.Series({"start": -1,  "match": -1 }))
# matche
# def PSDP(train_data,test_data,intereval):
#     if not (isinstance(train_data,pd.DataFrame) or isinstance(test_data,pd.DataFrame)) :
#         print("类型错误...")
#     else:
#################################################################


