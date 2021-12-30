import pandas as pd


def plt_shap(root_file):
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')
    # root_file = 'AT'
    names = ['A.thaliana', 'H.sapiens', 'M.musculus']
    if root_file == 'AT':
        name = names[0]
    elif root_file == 'human':
        name = names[1]
    else:
        name = names[2]
    root_file = root_file + '.csv'
    data_shap = pd.read_csv(r'./data/shap_' +root_file )
    data_fscore =pd.read_csv(r'./data/fscore_' +root_file )
    data_mrmr = pd.read_csv(r'./data/mrmr_' +root_file)

    Font = {'size': 18, 'family': 'Times New Roman'}

    i = 2

    recall1 = data_mrmr.iloc[:,1]
    precision1 = data_mrmr.iloc[:,i]

    recall2 = data_fscore.iloc[:,1]
    precision2 = data_fscore.iloc[:,i]

    recall3 = data_shap.iloc[:,1]
    precision3 = data_shap.iloc[:,i]

    plt.figure(figsize=(8,6),dpi=600)


    average_precision1 = 19
    plt.plot(recall3, precision3, '--', label='SHAP', color='Red',linewidth=2)
    plt.plot(recall1, precision1, 'b', label='mRMR', color='Blue',linewidth=2)
    plt.plot(recall2, precision2, ':', label='F1-score', color='y',linewidth=2)


    # plt.plot(recall2, precision2, 'b', label='Old_SVM : %0.3f' % average_precision2, color='Blue')

    plt.legend(loc='lower right', prop=Font)

    # plt.xlim([-0.01, 1.0])
    # plt.ylim([0, 1.01])
    plt.xlabel('Feature dimension', Font)
    plt.ylabel('AUROC', Font)
    plt.tick_params(labelsize=15)
    plt.title(name,Font)
    plt.grid()
    plt.savefig('./plt_new/'+root_file + '_plt.' + mat, format=mat)

    del plt
    # plt.show()

if __name__ == '__main__':
 for mat in ['pdf', 'eps', 'png']:
    for i in ['AT', 'mouse', 'human']:
        plt_shap(i)