# !/usr/bin/python3
# -*- codeing = utf-8 -*-
# @Time : 5/21/2021 1:28 PM
# @Author : Liu
# @File : svm_shap.py
# @Software : PyCharm


root_file = r'.\method/' # 最后必须以 ’/‘ 结尾#
method_list = ['xgb','svm', 'rf]
ctrl_step_list =['nps']
ps_ctrl_list = [0]
best_shap_number = ['yes']

import sys
import numpy as np
import os

root_txt_save_dir = r'./txt_save/' # 存储txt文件
root_np_save_dir = r'./np_save/' # 存储np文件——最终结果

if not os.path.exists(root_txt_save_dir):
    os.mkdir(root_txt_save_dir)

if not os.path.exists(root_np_save_dir):
    os.mkdir(root_np_save_dir)

sys.path.append(root_file + 'get_shap_seled')
sys.path.append(root_file + 'get_data')

from get_all_method import get_all_method
from get_shap_sort import get_shap_sort
from get_data import get_data
import param_data as pad

X_train, X_test, y_train, y_test, seq_train, seq_test = get_data(2)
idx_sorted = get_shap_sort(pad.xlf['xgb'], X_train, y_train)  # 获得shap排序（降序）

def save_txt(filename,filedata):
    file = open(filename, mode='w')
    file.write(str(filedata))  # 将 写入 txt文件中
    file.close()

all_data = {}

for pad_value in method_list:
    all_data[pad_value] = {}
    contrl_all = 0
    if best_shap_number[0] == 'yes':
        get_return_ctrl_all = get_all_method(pad_value, contrl_step='all', ps_ctrl=0, best_feature_n=0,root_file=root_file)
        all_data[pad_value]['all'] = {0: get_return_ctrl_all,}
        now_name_temp = pad_value + '-all-0'
        print('{}完成...'.format(now_name_temp))
        print('{}结果如下：'.format(now_name_temp))
        print(get_return_ctrl_all)
        print('--------------------------------------------------------')
        save_txt(root_txt_save_dir+now_name_temp + '-py.txt', get_return_ctrl_all)
        idx_sorted_c = 0
        best_feature_n = get_return_ctrl_all['best_shap_number']
    elif best_shap_number[0] == 'no':
        print('自定义idx_sorted...')
        idx_sorted_c = idx_sorted
        best_feature_n = 0 - best_shap_number[1]
        if contrl_all == 0:
            get_return_ctrl_all = get_all_method(pad_value, contrl_step='all', ps_ctrl=0, best_feature_n=best_feature_n,  root_file=root_file,idx_sorted=idx_sorted_c)
            all_data[pad_value]['all'] = {0: get_return_ctrl_all, }
            now_name_temp = pad_value + '-all-0'
            print('{}完成...'.format(now_name_temp))
            print('{}结果如下：'.format(now_name_temp))
            print(get_return_ctrl_all)
            print('--------------------------------------------------------')
            save_txt(root_txt_save_dir + now_name_temp + '-py.txt', get_return_ctrl_all)
            contrl_all = contrl_all + 1

np.save(root_txt_save_dir+'all_for_data.txt',all_data)



############## get_return_all 说明 ####################
    # pad_value = 'rf'  # 这里决定本次运行使用什么算法  ！！！！ 重点 ！！！！
    # ps_ctrl = 2  # 这里决定用psdp\psnp ： 0=不执行这一项， 1=psdp 2=psnp 3=psdp+psnp 其它数字不运行 {这里是数字不是字符}
    # ps_file_name = '../psnp_psdp' # 这里定义psnp/psdp的算法文件所在的目录
    # contrl_step : all shap_ps nps ps_n_other(只有ps没有其它特征)
    # 当 contrl_step='all' 时，best_feature_n 将无效，