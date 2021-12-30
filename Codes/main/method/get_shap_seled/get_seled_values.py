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
import math


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

# def get_seled_values(xlf_init, param_grid, X_train, X_test, y_train, y_test):
#     scoreings = {'ACC':'accuracy','Rec':'recall','roc_auc':'roc_auc','AP':'average_precision'}
#     xlf = xlf_init
#     optimized_GBM = GridSearchCV(xlf, param_grid=param_grid, scoring=scoreings, cv=5, refit='roc_auc',verbose=1, return_train_score=True,n_jobs=-1)
#     optimized_GBM.fit(X_train, y_train)
#     cv_accuracy = optimized_GBM.cv_results_['mean_test_ACC'][optimized_GBM.best_index_]
#     cv_auprc = optimized_GBM.cv_results_['mean_test_AP'][optimized_GBM.best_index_]
#     #cv_f1 = optimized_GBM.cv_results_['mean_test_f1'][optimized_GBM.best_index_]
#     #cv_pre = optimized_GBM.cv_results_['mean_test_Pre'][optimized_GBM.best_index_]
#     cv_rec = optimized_GBM.cv_results_['mean_test_Rec'][optimized_GBM.best_index_]
#
#
#     print('最优特征_格点搜索完成....')
#     return_data = {'best_parameters':optimized_GBM.best_params_,'roc_auc':optimized_GBM.best_score_, 'auprc':cv_auprc, "accuracy":cv_accuracy,"recall":cv_rec}
#     return optimized_GBM, return_data

def get_seled_values(xlf_init, param_grid, X_train, X_test, y_train, y_test):
    print('最优特征_格点搜索开始...')
    scorings = {'AUPRC': 'average_precision', 'ACC': 'accuracy', 'prec': 'precision', 'recall': 'recall', 'AUROC': 'roc_auc', 'f1':'f1'}
    # scorings = {'ACC':'accuracy','Rec':'recall','roc_auc':'roc_auc','AP':'average_precision'}
    xlf = xlf_init
    optimized_GBM = GridSearchCV(xlf, param_grid=param_grid, scoring=scorings, cv=5, refit='AUROC',verbose=1, return_train_score=True,n_jobs=-1)
    optimized_GBM.fit(X_train, y_train)
    # cv_rec = optimized_GBM.cv_results_['mean_test_Rec'][optimized_GBM.best_index_]
    cv_accracy = optimized_GBM.cv_results_['mean_test_ACC'][optimized_GBM.best_index_]
    cv_auprc = optimized_GBM.cv_results_['mean_test_AUPRC'][optimized_GBM.best_index_]
    cv_precision = optimized_GBM.cv_results_['mean_test_prec'][optimized_GBM.best_index_]
    cv_recall = optimized_GBM.cv_results_['mean_test_recall'][optimized_GBM.best_index_]
    cv_auroc = optimized_GBM.cv_results_['mean_test_AUROC'][optimized_GBM.best_index_]
    cv_f1 = optimized_GBM.cv_results_['mean_test_f1'][optimized_GBM.best_index_]
    best_params = optimized_GBM.best_params_
    y_train_t = y_train.tolist() # 重点
    TP1 = y_train_t.count(1) * cv_recall
    FP1 = (TP1 / cv_precision) - TP1
    TN1 = y_train_t.count(0) - FP1
    FN1 = y_train_t.count(1) - TP1
    cv_MCC = float(TP1 * TN1 - FP1 * FN1) / math.sqrt(float(TP1 + FP1) * float(TP1 + FN1) * float(TN1 + FP1) * float(TN1 + FN1))
    cv_specificity = TN1 / (TN1 + FP1)
    cv_sensitivity = TP1/(TP1+FN1)
    # cv_f1 = 2 * TP1 / (2 * TP1 + FP1 + FN1)
    return_data = {}
    y_scores = optimized_GBM.predict(X_test)  # 这里使用已经选择过后的特征
    y_scores1 = optimized_GBM.predict_proba(X_test)

    return_data['best_params'] = best_params
    return_data['best_cv_scores'] = {'TN':TN1, 'FP':FP1, 'FN':FN1, 'TP':TP1, 'F1':cv_f1, 'RECALL':cv_recall, 'SPE':cv_specificity, 'SEN':cv_sensitivity, 'ACC':cv_accracy, 'MCC':cv_MCC,'AUPRC':cv_auprc, 'AUROC':cv_auroc }

    print('最优特征_格点搜索完成....')
    return optimized_GBM, return_data, y_scores, y_scores1