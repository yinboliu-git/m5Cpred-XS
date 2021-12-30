# !/usr/bin/python3
# -*- codeing = utf-8 -*-
# @Time : 5/18/2021 10:14 PM
# @Author : Liu
# @File : param_data.py
# @Software : PyCharm
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.model_selection import GridSearchCV
import matplotlib
matplotlib.use("Agg")
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
import matplotlib
matplotlib.use("Agg")
import xgboost as xgb
import pandas as pd


# 格点搜索参数

param_grid = {
    'rf': {
        'max_depth': [2, 4, 6,8, 10,12,14,1],
        'n_estimators': [1600, 1800, 2000, 2200, 2400,2600,2800],
    },
    'xgb': {
        'max_depth': [ 2, 4, 6,8, 10,12,14,16],
        'learning_rate': [0.005, 0.01, 0.02, 0.05, 0.1],
        'n_estimators': [ 1600, 1800, 2000, 2200, 2400,2600,2800],
    },
    'svm': {
        "kernel": ['rbf'],
        "gamma": [2 ** i for i in range(-13, -8)],
        "C": [2 ** i for i in range(5, 9)],
    },

}

# xlf_tmp = xgb.XGBClassifier(
#                         learning_rate=0.01,
#                         max_depth=16,
#                         n_estimators=1800,
#                         objective='binary:logistic',
#                         nthread=-1,
#                         gamma=0,
#                         min_child_weight=1,
#                         max_delta_step=0,
#                         subsample=0.85,
#                         colsample_bytree=0.7,
#                         colsample_bylevel=1,
#                         reg_alpha=0,
#                         reg_lambda=1,
#                         scale_pos_weight=1,
#                         seed=1440,
#                        missing=None)


# 初试最优参数
param_best = {'rf': {'max_depth': [14], 'n_estimators': [1000]}, 'xgb': {'learning_rate': [0.005], 'max_depth': [14], 'n_estimators': [1800]}, 'svm':{'C': [256], 'gamma': [0.000244140625], 'kernel': ['rbf']},}
# 初试算法及其参数

xlf = {'xgb': xgb.XGBClassifier(max_depth=14,
                                learning_rate=0.005,
                                n_estimators=1800,
                                objective='binary:logistic',
                                nthread=-1,
                                gamma=0,
                                min_child_weight=1,
                                max_delta_step=0,
                                subsample=0.85,
                                colsample_bytree=0.7,
                                colsample_bylevel=1,
                                reg_alpha=0,
                                reg_lambda=1,
                                scale_pos_weight=1,
                                seed=1440,
                                missing=None),

       'svm': svm.SVC(probability=True),
       'rf': RandomForestClassifier(), }
