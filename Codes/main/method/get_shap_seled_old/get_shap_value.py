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

file_name = (str(os.path.basename(sys.argv[0])).split('.'))[0]  # 获取本文件（执行文件）的名字

def get_shap_value(xlf_init, param_grid, idx_sorted,X_train, X_test, y_train, y_test):
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
