# !/usr/bin/python3
# -*- codeing = utf-8 -*-
# @Time : 4/25/2021 9:35 PM
# @Author : Liu
# @File : PDSP_FOR.py
# @Software : PyCharm

import numpy as np
import pandas as pd


def PSDP(train_data, test_data, train_label, intereval):
    # 将数据转化为series格式
    # 获取DNA长度
    # train_data = ["sd",'sdf']
    DNA_len = train_data[0].__len__()  # 默认train和test的dna大小相同

    # 获取行数，也就是DNA的数据个数
    train_row_number = train_data.__len__()
    test_row_number = test_data.__len__()

    # 统计0、1的个数
    counst_0_1 =pd.DataFrame(train_label).value_counts()
    # tow_dna
    two_dna = ['AA', 'AU', 'AC', 'AG',
                      'UA', 'UU', 'UC', 'UG',
                      'CA', 'CU', 'CC', 'CG',
                      'GA', 'GU', 'GC', 'GG']
    # print(two_dna)
    #  print(str(train_data.iloc[1])[1])

    # 定义存储每个tow_dna在基因段中按位置出现的个数总数:A
    num_true =  np.zeros((16, DNA_len-intereval-1), dtype=int)
    num_false = np.zeros((16, DNA_len-intereval-1), dtype=int)

    for i in range(0, DNA_len-intereval-1):
        for j in range(0, 16):
            for k in range(0, train_row_number):
                if train_label[k] == 1:
                    if train_data[k][i] == two_dna[j][0] and train_data[k][i+intereval+1] == two_dna[j][1] :
                        num_true[j,i] = num_true[j,i]+1
                if train_label[k] == 0:
                    if train_data[k][i] == two_dna[j][0] and train_data[k][i+intereval+1] == two_dna[j][1] :
                        num_false[j,i] = num_false[j,i]+1

    freq = num_true/counst_0_1[1] - num_false/counst_0_1[0]

    # 定义返回值
    train_psdp = np.zeros((train_row_number, DNA_len - intereval - 1), dtype=float)
    test_psdp = np.zeros((test_row_number, DNA_len - intereval - 1), dtype=float)

    # 分别对train,test数据进行赋值freq
    for i in range(0, DNA_len- intereval - 1):
        for j in range(0, 16):
            for k in range(0, train_row_number):
                if train_data[k][i] == two_dna[j][0] and train_data[k][i+intereval+1] == two_dna[j][1] :
                    train_psdp[k,i] = freq[j,i]
            for k in range(0,test_row_number):
                if test_data[k][i] == two_dna[j][0] and test_data[k][i+intereval+1] == two_dna[j][1] :
                    test_psdp[k,i] = freq[j,i]

    return train_psdp, test_psdp



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


