#!/opt/share/bin/anaconda3/bin python
# coding: utf-8
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.model_selection import GridSearchCV
import matplotlib
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

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.metrics import roc_curve
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_curve, auc
from sklearn.metrics import make_scorer
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
import xgboost as xgb
import pandas as pd
import shap
from Bio import SeqIO
import sys



def get_seled_psdp_psnp( X_train, X_test, y_train, y_test, seq_train, seq_test,get_method, param_grid,ps_filename, contrl=3):# contrl 1：psdp,2:psnp,3:psdp+psnp
    sys.path.append(ps_filename)  # 这个目录随着服务器的改变需要改变
    print('加入psdp_psnp格点搜索开始...')
    print('本次搜索格点如下：')
    print(param_grid)
    if get_method == 'xgb':
        print('正在进行{}...'.format(get_method))
        optimized_GBM, return_data_xtrain, y_scores, y_scores1 = get_xgb_psdp_psnp(X_train, X_test, y_train, y_test, seq_train, seq_test, param_grid, contrl)
    elif get_method == 'rf':
        print('正在进行{}...'.format(get_method))
        optimized_GBM, return_data_xtrain, y_scores, y_scores1 = get_rf_psdp_psnp(X_train, X_test, y_train, y_test, seq_train, seq_test, param_grid, contrl)
    elif get_method == 'svm':
        print('正在进行{}...'.format(get_method))
        optimized_GBM, return_data_xtrain, y_scores, y_scores1 = get_svm_psdp_psnp(X_train, X_test, y_train, y_test, seq_train, seq_test, param_grid, contrl)
    return_data_xtrain_root = {}
    best_cvscores = return_data_xtrain['best_cv_scores']
    return_data_xtrain_root['best_cv_scores'] = {'TN':best_cvscores[0], 'FP':best_cvscores[1], 'FN':best_cvscores[2], 'TP':best_cvscores[3], 'F1':best_cvscores[4], 'RECALL':best_cvscores[5], 'SPE':best_cvscores[6], 'SEN':best_cvscores[7], 'ACC':best_cvscores[8], 'MCC':best_cvscores[9],'AUPRC':best_cvscores[10], 'AUROC':best_cvscores[11] }
    return_data_xtrain.pop('best_cv_scores')
    return_data_xtrain_root['best_params'] = return_data_xtrain
    return optimized_GBM, return_data_xtrain_root, y_scores, y_scores1


def pr_auc_score(label, prob):
    precision, recall, _thresholds = precision_recall_curve(label, prob)
    area = auc(recall, precision)
    return area


def my_score(y_true, y_pred, y_proba):
    TN, FP, FN, TP = metrics.confusion_matrix(y_true, y_pred).ravel()
    acc = metrics.accuracy_score(y_true, y_pred)
    MCC = metrics.matthews_corrcoef(y_true, y_pred)
    auroc = metrics.roc_auc_score(y_true, y_proba[:, 1])
    precision, recall, _thresholds = metrics.precision_recall_curve(y_true, y_proba[:, 1])
    auprc = metrics.auc(recall, precision)
    Sensitivity = TP / (TP + FN)
    Specificity = TN / (TN + FP)
    # ACC = (TP + TN) / (TP + FP + FN + TN)
    # Precision = TP / (TP + FP)
    F1Score = 2 * TP / (2 * TP + FP + FN)
    recall = TP / (TP + FN)
    # return_data = {'TN':TN, 'FP':FP, 'FN':FN, 'TP':TP, 'F1':F1Score, 'RECALL':recall, 'SPE':Specificity, 'SEN':Sensitivity, 'ACC':ACC, 'MCC':MCC,'AUPRC':area, 'AUROC':area1 }
    return TN, FP, FN, TP, F1Score, recall, Specificity, Sensitivity, acc, MCC, auroc, auprc

def my_score_2(y_true, y_pred, y_proba):
    TN, FP, FN, TP = metrics.confusion_matrix(y_true, y_pred).ravel()
    acc = metrics.accuracy_score(y_true, y_pred)
    MCC = metrics.matthews_corrcoef(y_true, y_pred)
    auroc = metrics.roc_auc_score(y_true, y_proba[:, 1])
    precision, recall, _thresholds = metrics.precision_recall_curve(y_true, y_proba[:, 1])
    auprc = metrics.auc(recall, precision)
    return TN, FP, FN, TP, acc, MCC, auroc, auprc


aucprc = make_scorer(pr_auc_score)


def get_rf_psdp_psnp(X_train, X_test, y_train, y_test, seq_train, seq_test, param_grid, contrl):  # contrl 1：psdp,2:psnp,3:psdp+psnp
    from PSDP.PSDP import PSDP
    from PSNP.PSNP import PSNP
    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    max_auroc = -1.0
    ########### 例子 ##########
    # 'rf': {
    #     'max_depth': [6, 8, 10, 12, 14, 16],
    #     'n_estimators': [1000, 1200, 1400, 1600, 1800],
    # }
    for md in param_grid['max_depth']:
        for nesti in param_grid['n_estimators']:
            xlf = RandomForestClassifier(max_depth=md, n_estimators=nesti)
            cvscores = []
            for tr_idx, val_idx in kfold.split(X_train, y_train):
                tr_X_tmp, val_X_tmp, tr_y, val_y = X_train[tr_idx], X_train[val_idx], y_train[tr_idx], y_train[val_idx]
                seq_tr_tmp, seq_val_tmp = seq_train[tr_idx], seq_train[val_idx]

                # 下面根据控制条件 设置 ！！！！
                if contrl == 1:
                    tr_psdp_tmp, val_psdp_tmp = PSDP(seq_tr_tmp, seq_val_tmp, tr_y, 0)
                    tr_X_tmp1 = np.concatenate((tr_X_tmp, tr_psdp_tmp), axis=1)
                    val_X_tmp1 = np.concatenate((val_X_tmp, val_psdp_tmp), axis=1)
                elif contrl == 2:
                    tr_psnp_tmp, val_psnp_tmp = PSNP(seq_tr_tmp, seq_val_tmp, tr_y)
                    tr_X_tmp1 = np.concatenate((tr_X_tmp, tr_psnp_tmp), axis=1)
                    val_X_tmp1 = np.concatenate((val_X_tmp, val_psnp_tmp), axis=1)
                elif contrl == 3:
                    tr_psdp_tmp, val_psdp_tmp = PSDP(seq_tr_tmp, seq_val_tmp, tr_y, 0)
                    tr_psnp_tmp, val_psnp_tmp = PSNP(seq_tr_tmp, seq_val_tmp, tr_y)
                    tr_X_tmp1 = np.concatenate((tr_X_tmp, tr_psnp_tmp, tr_psdp_tmp), axis=1)
                    val_X_tmp1 = np.concatenate((val_X_tmp, val_psnp_tmp, val_psdp_tmp), axis=1)

                xlf.fit(tr_X_tmp1, tr_y)
                val_score1 = xlf.predict(val_X_tmp1)
                val_score2 = xlf.predict_proba(val_X_tmp1)
                scores = my_score(val_y, val_score1, val_score2)
                cvscores.append(scores)
            mcvscore = np.mean(cvscores, axis=0)
            if mcvscore[10] > max_auroc:
                best_md = md
                best_nesti = nesti
                best_cvscores = mcvscore
                max_auroc = mcvscore[10]
        print('格点搜索{}完成...'.format(md))
    print('格点搜索以完全部完成...')

    return_data_xtrain = {'best_md':best_md, 'best_nesti':best_nesti, 'best_cv_scores':best_cvscores}
    #  best_cvscores : TN,FP,FN,TP,acc,MCC,auroc,auprc
    if contrl == 1:
        train_psdp, test_psdp = PSDP(seq_train, seq_test, y_train, 0)
        X_train_new = np.concatenate((X_train, train_psdp), axis=1)
        X_test_new = np.concatenate((X_test, test_psdp), axis=1)
    elif contrl == 2:
        train_psnp, test_psnp = PSNP(seq_train, seq_test, y_train)
        X_train_new = np.concatenate((X_train, train_psnp), axis=1)
        X_test_new = np.concatenate((X_test, test_psnp), axis=1)
    elif contrl == 3:
        train_psdp, test_psdp = PSDP(seq_train, seq_test, y_train, 0)
        train_psnp, test_psnp = PSNP(seq_train, seq_test, y_train)
        X_train_new = np.concatenate((X_train, train_psnp, train_psdp), axis=1)
        X_test_new = np.concatenate((X_test, test_psnp, test_psdp), axis=1)

    xlf = RandomForestClassifier(max_depth=best_md, n_estimators=best_nesti)
    xlf.fit(X_train_new, y_train)
    y_scores = xlf.predict(X_test_new)
    y_scores1 = xlf.predict_proba(X_test_new)
    print('rf与格点搜索完成...')
    # 注意，这里不再进行独立测试集各个参数的计算，在画图函数里进行计算
    return xlf, return_data_xtrain, y_scores, y_scores1


def get_svm_psdp_psnp(X_train, X_test, y_train, y_test, seq_train, seq_test, param_grid, contrl):  # contrl 1：psdp,2:psnp,3:psdp+psnp
    from PSDP.PSDP import PSDP
    from PSNP.PSNP import PSNP
    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    max_auroc = -1.0
    ########### 例子 ##########
    # 'svm': {
    #     "kernel": ['rbf'],
    #     "gamma": [2 ** i for i in range(-7, -3)],
    #     "C": [2 ** i for i in range(-1, 3)]
    # },
    for svm_g in param_grid['gamma']:
        for svm_c in param_grid['C']:
            #xlf = svm.SVC(C=svm_c, kernel='rbf', gamma=svm_g, probability=True)
            xlf = svm.SVC(C=svm_c, kernel='rbf', gamma=svm_g, probability=True)
            cvscores = []
            for tr_idx, val_idx in kfold.split(X_train, y_train):
                tr_X_tmp, val_X_tmp, tr_y, val_y = X_train[tr_idx], X_train[val_idx], y_train[tr_idx], y_train[val_idx]
                seq_tr_tmp, seq_val_tmp = seq_train[tr_idx], seq_train[val_idx]
                # 下面根据控制条件 设置 ！！！！
                if contrl == 1:
                    tr_psdp_tmp, val_psdp_tmp = PSDP(seq_tr_tmp, seq_val_tmp, tr_y, 0)
                    tr_X_tmp1 = np.concatenate((tr_X_tmp, tr_psdp_tmp), axis=1)
                    val_X_tmp1 = np.concatenate((val_X_tmp, val_psdp_tmp), axis=1)
                elif contrl == 2:
                    tr_psnp_tmp, val_psnp_tmp = PSNP(seq_tr_tmp, seq_val_tmp, tr_y)
                    tr_X_tmp1 = np.concatenate((tr_X_tmp, tr_psnp_tmp), axis=1)
                    val_X_tmp1 = np.concatenate((val_X_tmp, val_psnp_tmp), axis=1)
                elif contrl == 3:
                    tr_psdp_tmp, val_psdp_tmp = PSDP(seq_tr_tmp, seq_val_tmp, tr_y, 0)
                    tr_psnp_tmp, val_psnp_tmp = PSNP(seq_tr_tmp, seq_val_tmp, tr_y)
                    tr_X_tmp1 = np.concatenate((tr_X_tmp, tr_psnp_tmp, tr_psdp_tmp), axis=1)
                    val_X_tmp1 = np.concatenate((val_X_tmp, val_psnp_tmp, val_psdp_tmp), axis=1)

                xlf.fit(tr_X_tmp1, tr_y)
                val_score1 = xlf.predict(val_X_tmp1)
                val_score2 = xlf.predict_proba(val_X_tmp1)
                scores = my_score(val_y, val_score1, val_score2)
                cvscores.append(scores)
            mcvscore = np.mean(cvscores, axis=0)
            if mcvscore[10] > max_auroc:
                best_svm_g = svm_g
                best_svm_c = svm_c
                best_cvscores = mcvscore
                max_auroc = mcvscore[10]
        print('格点搜索{}完成...'.format(svm_g))
    print('格点搜索以完全部完成...')
    return_data_xtrain = {'best_svm_g':best_svm_g, 'best_svm_c':best_svm_c, 'best_cv_scores':best_cvscores}
    #  best_cvscores : TN,FP,FN,TP,acc,MCC,auroc,auprc
    if contrl == 1:
        train_psdp, test_psdp = PSDP(seq_train, seq_test, y_train, 0)
        X_train_new = np.concatenate((X_train, train_psdp), axis=1)
        X_test_new = np.concatenate((X_test, test_psdp), axis=1)
    elif contrl == 2:
        train_psnp, test_psnp = PSNP(seq_train, seq_test, y_train)
        X_train_new = np.concatenate((X_train, train_psnp), axis=1)
        X_test_new = np.concatenate((X_test, test_psnp), axis=1)
    elif contrl == 3:
        train_psdp, test_psdp = PSDP(seq_train, seq_test, y_train, 0)
        train_psnp, test_psnp = PSNP(seq_train, seq_test, y_train)
        X_train_new = np.concatenate((X_train, train_psnp, train_psdp), axis=1)
        X_test_new = np.concatenate((X_test, test_psnp, test_psdp), axis=1)

    xlf = svm.SVC(C=best_svm_c, kernel='rbf', gamma=best_svm_g, probability=True)
    xlf.fit(X_train_new, y_train)
    y_scores = xlf.predict(X_test_new)
    y_scores1 = xlf.predict_proba(X_test_new)

    print('svm与格点搜索完成...')
    return xlf,return_data_xtrain, y_scores, y_scores1

def get_xgb_psdp_psnp(X_train, X_test, y_train, y_test, seq_train, seq_test, param_grid,contrl): # contrl 1：psdp,2:psnp,3:psdp+psnp
    from PSDP.PSDP import PSDP
    from PSNP.PSNP import PSNP
    kfold = StratifiedKFold(n_splits = 5, shuffle= True, random_state=42)
    max_auroc = -1.0
    ########### 例子 ##########
    # 'xgb': {
    # 'max_depth': [2,4,6,8],
    # 'learning_rate': [0.005,0.01, 0.02, 0.05, 0.1],
    # 'n_estimators': [2000,2200,2400,2600,2800,3000],
    # },
    for md in param_grid['max_depth']:
        for lr in param_grid['learning_rate']:
            for nesti in param_grid['n_estimators']:
                xlf = xgb.XGBClassifier(max_depth=md,
                        learning_rate=lr,
                        n_estimators=nesti,
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
                       missing=None)
                cvscores = []
                for tr_idx, val_idx in kfold.split(X_train,y_train):
                    tr_X_tmp, val_X_tmp, tr_y, val_y = X_train[tr_idx], X_train[val_idx], y_train[tr_idx], y_train[val_idx]
                    seq_tr_tmp, seq_val_tmp = seq_train[tr_idx], seq_train[val_idx]

                    # 下面根据控制条件 设置 ！！！！
                    if contrl == 1:
                        tr_psdp_tmp, val_psdp_tmp = PSDP(seq_tr_tmp, seq_val_tmp, tr_y, 0)
                        tr_X_tmp1 = np.concatenate((tr_X_tmp, tr_psdp_tmp), axis=1)
                        val_X_tmp1 = np.concatenate((val_X_tmp,  val_psdp_tmp), axis=1)
                    elif contrl ==2 :
                        tr_psnp_tmp, val_psnp_tmp = PSNP(seq_tr_tmp, seq_val_tmp, tr_y)
                        tr_X_tmp1 = np.concatenate((tr_X_tmp, tr_psnp_tmp), axis=1)
                        val_X_tmp1 = np.concatenate((val_X_tmp, val_psnp_tmp), axis=1)
                    elif contrl == 3 :
                        tr_psdp_tmp, val_psdp_tmp = PSDP(seq_tr_tmp,seq_val_tmp,tr_y,0)
                        tr_psnp_tmp, val_psnp_tmp = PSNP(seq_tr_tmp,seq_val_tmp,tr_y)
                        tr_X_tmp1 = np.concatenate((tr_X_tmp,tr_psnp_tmp,tr_psdp_tmp),axis=1)
                        val_X_tmp1 = np.concatenate((val_X_tmp,val_psnp_tmp,val_psdp_tmp),axis=1)

                    xlf.fit(tr_X_tmp1,tr_y)
                    val_score1 = xlf.predict(val_X_tmp1)
                    val_score2 = xlf.predict_proba(val_X_tmp1)
                    scores = my_score(val_y,val_score1,val_score2)
                    cvscores.append(scores)
                mcvscore = np.mean(cvscores,axis=0)
                if mcvscore[10] > max_auroc:
                    best_md = md
                    best_lr = lr
                    best_nesti = nesti
                    best_cvscores = mcvscore
                    max_auroc = mcvscore[10]
        print('格点搜索{}完成...'.format(md))
    print('格点搜索以完全部完成...')
    return_data_xtrain = {'best_md':best_md, 'best_lr':best_lr, 'best_nesti':best_nesti,  'best_cv_scores':best_cvscores}
     #  best_cvscores : TN,FP,FN,TP,acc,MCC,auroc,auprc
    if contrl == 1:
        train_psdp,test_psdp = PSDP(seq_train, seq_test, y_train,0)
        X_train_new = np.concatenate((X_train,train_psdp),axis=1)
        X_test_new = np.concatenate((X_test,test_psdp),axis=1)
    elif contrl ==2 :
        train_psnp,test_psnp = PSNP(seq_train, seq_test, y_train)
        X_train_new = np.concatenate((X_train,train_psnp),axis=1)
        X_test_new = np.concatenate((X_test,test_psnp),axis=1)
    elif contrl == 3:
        train_psdp,test_psdp = PSDP(seq_train, seq_test, y_train,0)
        train_psnp,test_psnp = PSNP(seq_train, seq_test, y_train)
        X_train_new = np.concatenate((X_train,train_psnp,train_psdp),axis=1)
        X_test_new = np.concatenate((X_test,test_psnp,test_psdp),axis=1)


    xlf = xgb.XGBClassifier(max_depth=best_md,
            learning_rate=best_lr,
            n_estimators=best_nesti,
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
           missing=None)
    xlf.fit(X_train_new,y_train)
    y_scores = xlf.predict(X_test_new)
    y_scores1 = xlf.predict_proba(X_test_new)
    print('xgb与格点搜索完成...')
    # 这里不再进行独立测试集的各种数据的计算
    return xlf, return_data_xtrain, y_scores, y_scores1