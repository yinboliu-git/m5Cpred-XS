#!/opt/share/bin/anaconda3/bin python
# coding: utf-8
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
import matplotlib
#!/opt/share/bin/anaconda3/bin python
# coding: utf-8
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.model_selection import GridSearchCV
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
import matplotlib
matplotlib.use("Agg")
import sys
import os
# from ..get_data import param_data as pad # 运行时请把这行注释
import os
import sys
file_name_init = (str(os.path.basename(sys.argv[0])).split('.'))[0]  # 获取本文件（执行文件）的名字

def get_all_method(pad_value,contrl_step = 'all',ps_ctrl = 0, best_feature_n = 0, root_file = '../') :
    # pad_value 就是 method
    # pad_value = 'rf'  # 这里决定本次运行使用什么算法  ！！！！ 重点 ！！！！
    # ps_ctrl = 2  # 这里决定用psdp\psnp ：0 = 不进行这一步 1=psdp 2=psnp 3=psdp+psnp 其它数字不运行 {这里是数字不是字符}
    # ps_file_name = '../psnp_psdp' # 这里定义psnp/psdp的算法文件所在的目录
    # contrl_step : all shap_ps(shap_ps = best_shap_ps,最优特征+ps) nps ps_n_other(只有ps没有其它特征)
    all_param = [pad_value,contrl_step ,ps_ctrl , best_feature_n , root_file ]
    print('\n初始化运行，参数如下:')
    print(all_param)
    get_data_file = root_file + 'get_data'
    get_shap_seled_file = root_file + 'get_shap_seled'
    ps_file_name = root_file + 'psnp_psdp'
    return_data_all = {} # !!!!!!!!!!!!! 还未完全确定
    sys.path.append(get_data_file)
    sys.path.append(get_shap_seled_file)
    # 输入函数

    return_data_all['my_param'] = {}
    return_data_all['my_param']['method'] = pad_value
    return_data_all['my_param']['contrl_step'] = contrl_step
    return_data_all['my_param']['ps_ctrl'] = ps_ctrl

    from get_data import get_data
    import param_data as pad

    from get_shap_sort import get_shap_sort
    from get_shap_value import get_shap_value
    from get_seled_values import get_seled_values
    from get_seled_psdp_psnp import get_seled_psdp_psnp
    # 输出数据函数
    from print_data import print_data

    X_train, X_test, y_train, y_test, seq_train, seq_test = get_data(2)  # 获得数据 参数1=捕获的ps数据，2=获得ps数据

    # 每次运行只需要更改一下代码：(暂不用改）
    shap_xlf_init = pad.xlf['xgb']  # 这里默认使用xgb
    xlf_init_train = pad.xlf[pad_value] # 训练用算法 - 训练 - 训练后格点搜索
    param_grid_train = pad.param_best[pad_value]  # 选择shap训练用参数 使用LM.log里的最优参数
    param_grid_seled = pad.param_grid[pad_value]  # shap完成后格点搜索用参数
    if not(contrl_step in ['all' ,'nps' ,'ps_n_other', 'shap_ps', 'best_shap_ps']):
        print('contrl_step输入错误...')
        return '错误..   请检查contrl_step是否错误...'

    ########### 特征选择时的构造 shap特征排序 ######

    if contrl_step != 'ps_n_other':
            idx_sorted = get_shap_sort(shap_xlf_init, X_train, y_train)  # 获得shap排序（降序）

    if contrl_step == 'all':
        ############# shap选择过后进行train训练 返回最优特征 ##########
        best_value = get_shap_value(xlf_init_train, param_grid_train, idx_sorted, X_train, X_test, y_train, y_test)
        # best_value 就是best_shap_number
        print(best_value)
        print(best_value.iloc[0,0])
        best_feature_n = 0 - best_value.iloc[0, 0]

    return_data_all['best_shap_number'] = best_feature_n

    if contrl_step != 'ps_n_other':
        X_train = X_train[:, idx_sorted[best_feature_n:]]
        X_test = X_test[:, idx_sorted[best_feature_n:]]
        print('从特征中选择数组完成....')
    elif contrl_step == 'ps_n_other':
        X_train = np.zeros((X_train.shape[0],1))
        X_test = np.zeros((X_test.shape[0],1))
        print('自动创建0数组完成....')

    if contrl_step == 'all' or contrl_step == 'nps':
        ####### seled进行格点搜索 ########## //下面对最优特征进行格点搜索
        optimized_GBM, return_data, y_scores, y_scores1 = get_seled_values(xlf_init_train, param_grid_seled, X_train, X_test, y_train, y_test)
        print(return_data)
        return_data_all['shap_gs_seled_values'] = return_data

        ##### 画图 - 独立测试数据 ##########
        # y_scores = optimized_GBM.predict(X_test)  # 这里使用已经选择过后的特征
        # y_scores1 = optimized_GBM.predict_proba(X_test)
        id_scores_data = print_data(y_test, y_scores, y_scores1, name_add = 'seled_id')  # 独立测试集的 返回最优结论值
        print(id_scores_data)
        return_data_all['shap_gs_seled_values_idx'] = id_scores_data # 独立测试集
        return_data_all['gs_nps_y_scores'] = [y_scores]
        return_data_all['gs_nps_y_scores1'] = [y_scores1]

    if (contrl_step in ['all', 'shap_ps' ,'best_shap_ps', 'ps_n_other']) and ps_ctrl > 0 :
        ##### 加入psnp_psdp 进行格点搜索 #######
        get_method = pad_value
        param_grid = pad.param_grid[pad_value]
        ps_filename = ps_file_name
        optimized_GBM, return_data_xtrain, y_scores, y_scores1 = get_seled_psdp_psnp(X_train, X_test, y_train, y_test, seq_train, seq_test, get_method, param_grid,ps_filename, contrl=ps_ctrl)
        print(return_data_xtrain)
        return_data_name = str(ps_ctrl)+contrl_step + '_gs'
        return_data_all[return_data_name] = return_data_xtrain
        return_data_all[return_data_name]
        return_data_name_idx = return_data_name + '_idx'
        #### 画图 ###
        id_scores_data = print_data(y_test, y_scores, y_scores1, name_add=str(ps_ctrl)+contrl_step)  # 独立测试集的 返回最优结论值
        print(id_scores_data)
        return_data_all[return_data_name_idx] = id_scores_data
        return_data_all['gs_ps_y_scores'] = [y_scores]
        return_data_all['gs_ps_y_scores1'] = [y_scores1]
    np.save(pad_value+'_'+file_name_init+'_rd_all_' + str(ps_ctrl) + contrl_step + '.txt', return_data_all)

    return return_data_all
