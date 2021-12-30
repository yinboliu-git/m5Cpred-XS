# !/usr/bin/python3
# -*- codeing = utf-8 -*-
# @Time : 5/18/2021 8:32 PM
# @Author : Liu
# @File : shap.py
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

def get_shap_sort(xlf_init, X_train, y_train):
    xlf = xlf_init
    xlf.fit(X_train, y_train)
    print('shap特征排序构造完成...')
    explainer = shap.TreeExplainer(xlf)
    shap_values = explainer.shap_values(X_train)

    xlf_shap_import = np.mean(abs(shap_values),axis=0) #this is the shap importance for each feature based on all train data
    #sort the shap importance
    idx_sorted = np.argsort(xlf_shap_import) #this is ascend
    print('shap特征选择排序完成....')
    return idx_sorted