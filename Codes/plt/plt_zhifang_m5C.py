import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
def count_number():
    max_number = [164, 312, 353, 476, 612, 708, 808]

    data = pd.read_excel("./data_tj.xlsx")
    count_ = [[],[],[]]
    all_count = []
    c = 0
    for one_site, j in enumerate(['H', 'M', "A"]):
        all_count.append([])
        for c,_ in enumerate(max_number):
            all_count[one_site].append(0)

        for ii in range(data[j].__len__()):
            for site, word in enumerate(max_number):
                if data[j][ii] < word:
                    all_count[one_site][site] = all_count[one_site][site] + 1
                    break
    print(all_count)

    return all_count


def plt_zf(count):
    # 设置画布的大小
 for mat in ['png', 'pdf', 'eps']:
  for i,mm in enumerate(['H.sapiens', 'M.musculus','A.thaliana']):
    plt.figure(figsize=(10,5),dpi=600)
    max_number = [0,164, 312, 353, 476, 612, 708, 808]
    # 输入统计数据
    name = ['Binary ', 'ENAC   ','ANF     ', 'NCP      ','SCPseDNC','CKSNAP ', 'FastText '] # 在这里设置名字
    for site in range(max_number.__len__()-1):
        name[site] = name[site] + '\n(' + str(max_number[site]) + '~' + str(max_number[site+1]-1) + ')'

    name.reverse()
    GLy_pseaac = count[i]
    GLy_pseaac.reverse()# 交叉验证在这里设置高度
    # BPB_Glysite = [0.569,0.616,0.603,0.191,0.592,0.643]
    # GLy_pseaac = [0.69	0.55	0.605263158	0.242387149	0.62	0.679275],# 独立测试在这里设置高度
    # BPB_Glysite = [0.569,0.616,0.603,0.191,0.592,0.643]
    bar_width = 0.9 # 条形宽度
    index_a = np.arange(len(name))
    # index_b = index_a + bar_width

    y = [GLy_pseaac]
    x = [index_a]
    plt.grid(zorder=0)

    # 使用4次 bar 函数画出两组条形图
    plt.barh(index_a,GLy_pseaac,height=bar_width, color='lawngreen',edgecolor='black',zorder=10)
    # plt.bar(index_b, height=BPB_Glysite, width=bar_width, color='y', label='BERTprot-CLS-CNN',zorder=10)
    # plt.barh(waters, buy_number)  # 横放条形图函数 barh
    # plt.bar(index_c, height=GlyNN, width=bar_width, color='c', label='GlyNN',zorder=10)
    # plt.bar(index_d, height=my_model, width=bar_width, color='r', label='My_Model',zorder=10)


    # plt.legend()  # 显示图例
    plt.yticks(index_a, name)  # 将横坐标显示在每个坐标的中心位置


    # 给柱状图添加高度
    Font_1 = {
        'size': 10,
        'weight': 'bold'
    }
    for x_ind,y_ind in zip(x,y):
            for i in range(len(x_ind)):
                    yy1 = x_ind[i]
                    xx1= y_ind[i]
                    plt.text(xx1+0.2, yy1, ('%d' % xx1),fontdict=Font_1, ha='center', va='bottom',zorder=11)

    Font_2 = {
                'size': 15,
            'weight': 'bold'
        }

    plt.xlabel('Count', Font_2)  # 横坐标轴标题
    plt.ylabel('Feature encondings', Font_2, zorder=12)  # 纵坐标轴标题
    plt.title(mm,Font_2)  # 图形标题
    ax = plt.gca()
    ax.spines['bottom'].set_linewidth(2);  ###设置底部坐标轴的粗细
    ax.spines['left'].set_linewidth(2);  ####设置左边坐标轴的粗细
    ax.spines['left'].set_zorder(11);
    # plt.show()
    plt.savefig('./m5cimg/' +mm + '.' + mat,format=mat)


if __name__ == '__main__':
    count = count_number()
    plt_zf(count)