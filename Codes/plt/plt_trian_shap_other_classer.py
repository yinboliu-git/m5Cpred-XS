from sklearn import metrics
import pylab as plt
import numpy as np


# 画roc svm - xgb - rf
def ks(y_predicted1, y_true1, y_predicted2, y_true2, y_predicted3, y_true3,name,mat):
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')
    
    names = ['A.thaliana', 'H.sapiens', 'M.musculus']
    root_file = name
    if root_file == 'AT':
        name = names[0]
    elif root_file == 'human':
        name = names[1]
    else:
        name = names[2]

    Font = {'size': 16,'weight':'bold', 'family': 'Times New Roman'}
    Font1 = {'size': 15, 'family': 'Times New Roman'}

    label1 = y_true1
    label2 = y_true2
    label3 = y_true3
    fpr1, tpr1, thres1 = metrics.roc_curve(label1, y_predicted1)
    fpr2, tpr2, thres2 = metrics.roc_curve(label2, y_predicted2)
    fpr3, tpr3, thres3 = metrics.roc_curve(label3, y_predicted3)
    roc_auc1 = metrics.auc(fpr1, tpr1)
    roc_auc2 = metrics.auc(fpr2, tpr2)
    roc_auc3 = metrics.auc(fpr3, tpr3)

    plt.figure(figsize=(8, 8), dpi=600)
    plt.plot(fpr1, tpr1, 'b', label='AUROC(XGBoost-SHAP) = %0.3f' % roc_auc1, color='Red')
    plt.plot(fpr2, tpr2, 'b', label='AUROC(SVM-SHAP) = %0.3f' % roc_auc2, color='k')
    plt.plot(fpr3, tpr3, 'b', label='AUROC(Rf-SHAP) = %0.3f' % roc_auc3, color='RoyalBlue')
    plt.legend(loc='lower right', prop=Font1)
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([-0.01, 1.01])
    plt.ylim([-0.01, 1.01])
    plt.ylabel('True Positive Rate', Font)
    plt.xlabel('False Positive Rate', Font)
    plt.grid()   
    plt.tick_params(labelsize=15)
    ax = plt.gca()
    ax.spines['bottom'].set_linewidth(2)
    ax.spines['left'].set_linewidth(2)
    plt.title(name,Font)
    plt.savefig('./img_shap/' +name+'-SHAP' + '.' + mat, format=mat)
    return abs(fpr1 - tpr1).max(), abs(fpr2 - tpr2).max(), abs(fpr3 - tpr3).max()

# roc no-shap shap
def ks_2(y_predicted1, y_true1, y_predicted2, y_true2,name,model,mat):
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')
    Font = {'size': 16,'weight':'bold', 'family': 'Times New Roman'}

    names = ['A.thaliana', 'H.sapiens', 'M.musculus']
    root_file = name
    if root_file == 'AT':
        name = names[0]
    elif root_file == 'human':
        name = names[1]
    else:
        name = names[2]

    label1 = y_true1
    label2 = y_true2

    fpr1, tpr1, thres1 = metrics.roc_curve(label1, y_predicted1)
    fpr2, tpr2, thres2 = metrics.roc_curve(label2, y_predicted2)

    roc_auc1 = metrics.auc(fpr1, tpr1)
    roc_auc2 = metrics.auc(fpr2, tpr2)

    plt.figure(figsize=(8, 8), dpi=600)
    plt.plot(fpr1, tpr1, '--', label='AUROC(SHAP) = %0.3f' % roc_auc1, color='Red',linewidth=2)
    plt.plot(fpr2, tpr2, ':', label='AUROC(No-SHAP) = %0.3f' % roc_auc2, color='Blue', linewidth=2)
    Font2 = {'size': 15, 'family': 'Times New Roman'}

    plt.legend(loc='lower right', prop=Font2)
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([-0.01, 1.01])
    plt.ylim([-0.01, 1.01])
    plt.ylabel('True Positive Rate', Font)
    plt.xlabel('False Positive Rate', Font)
    plt.tick_params(labelsize=15)
    plt.grid()
    plt.title(name+'-' + model,Font)
    plt.title(name+'-' + model,Font)
    
    ax = plt.gca()
    ax.spines['bottom'].set_linewidth(2)
    ax.spines['left'].set_linewidth(2)
    plt.savefig('./img/' +name+'_' + model + '.' + mat, format=mat)
    # plt.savefig('./plt_new/'+root_file + '_plt.' + mat, format=mat)
    return abs(fpr1 - tpr1).max(), abs(fpr2 - tpr2).max()

# 获取scores
def read_data(root_data_name):
    import pandas as pd
    import numpy as np
    import os
    file_self = os.getcwd()

    root_name = root_data_name
    os.chdir("./data_train/" + root_name)
    file_chdir = os.getcwd()
    filename_npy = []
    file_npy = []
    for root, dirs, files in os.walk(file_chdir):
        for file in files:
            print(1)
            if os.path.splitext(file)[-1] == '.npy':
                if 'no' in file.split('_'):
                    print(3)
                    filename_npy.append('no_' + file.split('_')[0])
                    file_npy.append(np.load(file, allow_pickle=True))
                else:
                    print(2)
                    filename_npy.append(file.split('_')[0])
                    file_npy.append(np.load(file, allow_pickle=True))

    # best_train_i = 0
    # best_test_i = 0
    # print(list(list(file_npy[0].item().values())[6]))
    print(file_npy[5])
    print()
    csvname_scores = {}
    csvname_ytest = {}
    for i in range(0, len(file_npy)):

        csvname_ytest[filename_npy[i]] = list(list(file_npy[i].item().values())[5])[0][:,1]
        csvname_scores[filename_npy[i]] = list(list(file_npy[i].item().values())[6])
    print(csvname_scores['xgb'])
    print(csvname_scores['no_xgb'])
    os.chdir(file_self)

    return csvname_scores,csvname_ytest

if __name__ == '__main__':
  for file in ['AT', 'human', 'mouse']:
    y_scores,ytest = read_data(file+'_data')
    name = file
    print()
    y_predicted1 = ytest['xgb']
    y_test1 = y_scores['xgb']
    y_predicted2 = ytest['svm']
    y_test2 = y_scores['svm']
    y_predicted3 = ytest['rf']
    y_test3 = y_scores['rf']

    y_test_no1 = ytest['no_xgb']
    y_predicted_no1 = y_scores['no_xgb']

    y_test_no2 = ytest['no_svm']
    y_predicted_no2 = y_scores['no_svm']

    y_test_no3 = ytest['no_rf']
    y_predicted_no3 = y_scores["no_rf"]

    # print(y_test)
    for mat in ['png','pdf','eps']:
        ks(y_predicted1, y_test1, y_predicted2, y_test2, y_predicted3,y_test3,name,mat)
