# !/usr/bin/python3
# -*- codeing = utf-8 -*-
# @Time : 5/21/2021 1:28 PM
# @Author : Liu
# @File : svm_shap.py
# @Software : PyCharm

import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.model_selection import GridSearchCV
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.metrics import roc_curve
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_curve,auc
from sklearn.metrics import make_scorer
from sklearn.model_selection import cross_val_score
import xgboost as xgb
import pandas as pd
import shap
import os
import sys


root_file = r'C:\Users\ybliu\Desktop\研究生工作\temp\AT\method_1/' # 最后必须以 ’/‘ 结尾#
method_list = ['xgb' ,'rf']
ctrl_step_list =['shap_ps', 'ps_n_other']
ps_ctrl_list = [1,2,3]
best_shap_number = ['no',228]  # 这里配置是否自动生成best_shap_number, 使用['yes'],不使用自动生成、用已知的['no',228],后面这个是已经计算好的

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
np.set_printoptions(threshold=np.inf) # 设置np输出为不省略格式
from get_all_method import get_all_method
from get_shap_sort import get_shap_sort
from get_data import get_data
import param_data as pad

X_train, X_test, y_train, y_test, seq_train, seq_test = get_data(2)
idx_sorted = get_shap_sort(pad.xlf['xgb'], X_train, y_train)  # 获得shap排序（降序）
print(idx_sorted)
def save_txt(filename,filedata):
    file = open(filename, mode='w')
    file.write(str(filedata))  # 将 写入 txt文件中
    file.close()

file_name = (str(os.path.basename(sys.argv[0])).split('.'))[0]  # 获取本文件（执行文件）的名字

def get_fscore_value(xlf_init, param_grid, idx_sorted,X_train, X_test, y_train, y_test):
    aucroc_on_train = []
    aucroc_on_test = []
    featn = []
    xlf_tmp = xlf_init
    for i in range(10,len(idx_sorted),10): # 请将最后一个数字改成10， 其它为测试用
        X_train_tmp = X_train[:,idx_sorted[i:]]
        X_test_tmp = X_test[:,idx_sorted[i:]]
        optimized_GBM = GridSearchCV(xlf_tmp, param_grid=param_grid, scoring='roc_auc', cv=5, refit=True,verbose=1, return_train_score=True,n_jobs=-1)
        optimized_GBM.fit(X_train_tmp, y_train)
        print("Best parameters:{}".format(optimized_GBM.best_params_))
        print("Test set roc_auc:{:.4f}".format(optimized_GBM.score(X_test_tmp,y_test)))
        print("Best roc_auc on train set:{:.4f}".format(optimized_GBM.best_score_))
        featn.append(len(idx_sorted)-i)
        aucroc_on_train.append(optimized_GBM.best_score_)
        aucroc_on_test.append(optimized_GBM.score(X_test_tmp,y_test))
        print('***************第{}特征完成**************'.format(i))

    newdf = pd.DataFrame({'featn':featn,'aucroc_on_train':aucroc_on_train,'aucroc_on_test':aucroc_on_test})
    newdf.to_csv('./'+file_name+ 'get_shap_value.csv')
    sort_df = newdf.sort_values(by='aucroc_on_train',ascending=False)
    print('最优特征选择完成...')
    return sort_df.iloc[0:1,:]


all_data = {}
for pad_value in method_list:
    contrl_all = 0  # 这行代码被我改变了
    all_data[pad_value] = {}
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

    # for contrl_step in ctrl_step_list:
    #     all_data[pad_value][contrl_step] = {}
    #     for ps_ctrl in ps_ctrl_list:
    #         get_return_all = get_all_method(pad_value,contrl_step = contrl_step, ps_ctrl=ps_ctrl,best_feature_n = best_feature_n, root_file=root_file, idx_sorted=idx_sorted_c)
    #         now_name_temp = pad_value+'-'+contrl_step+'-'+str(ps_ctrl)
    #         all_data[pad_value][contrl_step][ps_ctrl] = get_return_all
    #         print('{}完成...'.format(now_name_temp))
    #         print('{}结果如下：'.format(now_name_temp))
    #         print(get_return_all)
    #         print('--------------------------------------------------------')
    #         file_name_temp = now_name_temp+'-py.txt'
    #         save_txt(root_txt_save_dir+file_name_temp,get_return_all)

np.save(root_txt_save_dir+'all_for_data.txt',all_data)



############## get_return_all 说明 ####################
    # pad_value = 'rf'  # 这里决定本次运行使用什么算法  ！！！！ 重点 ！！！！
    # ps_ctrl = 2  # 这里决定用psdp\psnp ： 0=不执行这一项， 1=psdp 2=psnp 3=psdp+psnp 其它数字不运行 {这里是数字不是字符}
    # ps_file_name = '../psnp_psdp' # 这里定义psnp/psdp的算法文件所在的目录
    # contrl_step : all shap_ps nps ps_n_other(只有ps没有其它特征)
    # 当 contrl_step='all' 时，best_feature_n 将无效，