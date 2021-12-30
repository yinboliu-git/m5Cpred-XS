import sys
import numpy as np
import os

def save_txt(filename, filedata):
    file = open(filename, mode='w')
    file.write(str(filedata))  # 将 写入 txt文件中
    file.close()


def get_all_for(method_list, ctrl_step_list, ps_ctrl_list, root_file):

    sys.path.append(root_file + 'get_shap_seled')
    from get_all_method import get_all_method

    root_txt_save_dir = r'./txt_save/'  # 存储txt文件
    root_np_save_dir = r'./np_save/'  # 存储np文件——最终结果

    if not os.path.exists(root_txt_save_dir):
        os.mkdir(root_txt_save_dir)
    if not os.path.exists(root_np_save_dir):
        os.mkdir(root_np_save_dir)

    all_data = {}
    for pad_value in method_list:
        all_data[pad_value] = {}
        get_return_ctrl_all = get_all_method(pad_value, contrl_step='all', ps_ctrl=0, best_feature_n=0,
                                             root_file=root_file)
        all_data[pad_value]['all'] = {0: get_return_ctrl_all, }
        now_name_temp = pad_value + '-all-0'
        print('{}完成...'.format(now_name_temp))
        print('{}结果如下：'.format(now_name_temp))
        print(get_return_ctrl_all)
        print('--------------------------------------------------------')
        save_txt(root_txt_save_dir + now_name_temp + '-py.txt', get_return_ctrl_all)
        best_feature_n = get_return_ctrl_all['best_shap_number']

        for contrl_step in ctrl_step_list:
            all_data[pad_value][contrl_step] = {}
            for ps_ctrl in ps_ctrl_list:
                get_return_all = get_all_method(pad_value, contrl_step=contrl_step, ps_ctrl=ps_ctrl,
                                                best_feature_n=best_feature_n, root_file=root_file)
                now_name_temp = pad_value + '-' + contrl_step + '-' + str(ps_ctrl)
                all_data[pad_value][contrl_step][ps_ctrl] = get_return_all
                print('{}完成...'.format(now_name_temp))
                print('{}结果如下：'.format(now_name_temp))
                print(get_return_all)
                print('--------------------------------------------------------')
                file_name_temp = now_name_temp + '-py.txt'
                save_txt(root_txt_save_dir + file_name_temp, get_return_all)

    np.save(root_txt_save_dir + 'all_for_data.txt', all_data)
    return all_data

############## get_return_all 说明 ####################
# pad_value = 'rf'  # 这里决定本次运行使用什么算法  ！！！！ 重点 ！！！！
# ps_ctrl = 2  # 这里决定用psdp\psnp ： 0=不执行这一项， 1=psdp 2=psnp 3=psdp+psnp 其它数字不运行 {这里是数字不是字符}
# ps_file_name = '../psnp_psdp' # 这里定义psnp/psdp的算法文件所在的目录
# contrl_step : all shap_ps nps ps_n_other(只有ps没有其它特征)
# 当 contrl_step='all' 时，best_feature_n 将无效，