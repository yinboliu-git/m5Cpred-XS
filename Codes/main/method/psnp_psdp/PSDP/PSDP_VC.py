# !/usr/bin/python3
# -*- codeing = utf-8 -*-
# @Time : 4/25/2021 1:44 PM
# @Author : Liu
# @File : PSDP.py
# @Software : PyCharm

import numpy as np
import pandas as pd
import re

# 测试此模块是否导入成功的代码
A = 1


# 主函数
def PSDP(train_data, test_data, train_label, intereval):
    # 将数据转化为series格式
    try:
        if not isinstance(train_data, pd.Series) :
            train_data = pd.Series(train_data)
        if not isinstance(test_data, pd.Series) :
            test_data = pd.Series(test_data)
        if not isinstance(train_label, pd.Series) :
            train_label = pd.Series(train_label)
    except Exception as e:
        print("PSDP中发生错误，请检查您输入PSDP的数据格式...\n{}".format(e))

    # 获取DNA长度
    DNA_len_all = train_data.str.len()
    DNA_len = DNA_len_all.iloc[0]  # 默认train和test的dna大小相同

    # 获取行数，也就是DNA的数据个数
    train_row_number = train_data.size
    test_row_number = test_data.size

    # 统计0、1的个数
    counst_0_1 =train_label.value_counts()

    # tow_dna
    two_dna = pd.Series(['AA', 'AU', 'AC', 'AG',
                      'UA', 'UU', 'UC', 'UG',
                      'CA', 'CU', 'CC', 'CG',
                      'GA', 'GU', 'GC', 'GG'])

    # 定义存储每个tow_dna在基因段中按位置出现的个数总数:A
    num_true = pd.DataFrame(np.zeros((16, DNA_len-intereval-1), dtype=int))
    num_false = pd.DataFrame(np.zeros((16, DNA_len-intereval-1), dtype=int))

    # 生成匹配字符-正则表达式
    macth_str = '(' + two_dna.str[0] + '.{'+ str(intereval) +'}' + two_dna.str[1] + ')'

    # 个数统计开始：A
    myfind_1 = []
    myfind_0 = []
    for i in range(0, 16):
        # print(macth_str[i])
        myfind_1.append(myfindall(train_data[train_label == 1], macth_str[i]))
        num_true.loc[i] = myfind_1[i][myfind_1[i]['start']>=0]['start'].value_counts()  # 合计value
        myfind_0.append(myfindall(train_data[train_label == 0], macth_str[i]))
        num_false.loc[i] = myfind_0[i][myfind_0[i]['start'] >= 0]['start'].value_counts()   # 合计value
        print('正在处理{}/16条数据...'.format(i+1))
    print('处理完毕...')

    # 防止有空值出现
    num_true = num_true.fillna(0)
    num_false = num_false.fillna(0)

    # 计算频率
    freq = num_true/counst_0_1[1] - num_false/counst_0_1[0]

    # 定义返回值
    train_psdp = pd.DataFrame(np.zeros((train_row_number, DNA_len - intereval - 1), dtype=float))
    test_psdp = pd.DataFrame(np.zeros((test_row_number, DNA_len - intereval - 1), dtype=float))

    # 逻辑：设x位置出现了某基因two_dna,那么将freq(基因，位置)赋值给返回值
    for i in range(0, 16):
        for j in range(0, DNA_len-intereval-1):
            train_psdp.iloc[myfind_1[i][myfind_1[i]['start'] == j].index.tolist(), j] = freq.iloc[i, j]
            train_psdp.iloc[myfind_0[i][myfind_0[i]['start'] == j].index.tolist(), j] = freq.iloc[i, j]
            my_temp_find_test = myfindall(test_data, macth_str[i])
            test_psdp.iloc[(my_temp_find_test[my_temp_find_test['start'] == j].index.tolist()), j] = freq.iloc[i, j]
            # print(i,j)
        print('正在处理{}/16条数据...'.format(i+1))
    print('处理完毕...')

    return train_psdp, test_psdp

# 找到所有出现re_find的位置
def myfindall(series, re_find):
    reg = re.compile(re_find)
    matches = series.apply(lambda r: list(reg.finditer(r)))     # 此条语句速度超级慢
    # 构造dataframe,并且给空值赋值-1
    return_find = matches.explode().apply(lambda r: pd.Series({"start": r.span()[0],  "match": r.group(0) }) if isinstance(r,re.Match) else pd.Series({"start": -1,  "match": -1 }))    # 此条语句速度同样超级慢
    return return_find


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


