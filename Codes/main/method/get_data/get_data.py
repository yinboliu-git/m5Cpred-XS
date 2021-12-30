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
from Bio import SeqIO
def readfa(filename):
    seqs = []
    for seq_record in SeqIO.parse(filename,"fasta"):
        seqs.append(str(seq_record.seq))
    return seqs

def get_data(contrl = 1):
    pos_binary = pd.read_csv("../../AT6289_pos_binary.csv",header=None,usecols=[i for i in range(1,165)])
    neg_binary = pd.read_csv("../../AT6289_neg_binary.csv",header=None,usecols=[i for i in range(1,165)])
    pos_enac = pd.read_csv("../../AT6289_pos_ENAC.csv",header=None,usecols=[i for i in range(1,149)])
    pos_anf = pd.read_csv("../../AT6289_pos_ANF.csv",header=None,usecols=[i for i in range(1,42)])
    pos_ncp = pd.read_csv("../../AT6289_pos_NCP.csv",header=None,usecols=[i for i in range(1,124)])
    #pos_psednc = pd.read_csv("mouses-pos-5563-PseDNC.csv",header=None,usecols=[i for i in range(1,19)])
    pos_scpsednc = pd.read_csv("../../AT6289_pos_SCPseDNC.csv",header=None,usecols=[i for i in range(1,137)])
    #pos_kmer = pd.read_csv("mouse-pos-5563-Kmer.csv",header=None,usecols=[i for i in range(1,257)])
    pos_cksnap = pd.read_csv("../../AT6289_pos_CKSNAP.csv",header=None,usecols=[i for i in range(1,97)])
    pos_all = pd.concat([pos_binary,pos_enac,pos_anf,pos_ncp,pos_scpsednc,pos_cksnap],axis=1)
    neg_enac = pd.read_csv("../../AT6289_neg_ENAC.csv",header=None,usecols=[i for i in range(1,149)])
    neg_anf = pd.read_csv("../../AT6289_neg_ANF.csv",header=None,usecols=[i for i in range(1,42)])
    neg_ncp = pd.read_csv("../../AT6289_neg_NCP.csv",header=None,usecols=[i for i in range(1,124)])
    #neg_psednc = pd.read_csv("mouses-neg-5563-PseDNC.csv",header=None,usecols=[i for i in range(1,19)])
    neg_scpsednc = pd.read_csv("../../AT6289_neg_SCPseDNC.csv",header=None,usecols=[i for i in range(1,137)])
    #neg_kmer = pd.read_csv("mouse-neg-5563-Kmer.csv",header=None,usecols=[i for i in range(1,257)])
    neg_cksnap = pd.read_csv("../../AT6289_neg_CKSNAP.csv",header=None,usecols=[i for i in range(1,97)])
    neg_all = pd.concat([neg_binary,neg_enac,neg_anf,neg_ncp,neg_scpsednc,neg_cksnap],axis=1)
    posall=np.array(pos_all)
    #postrain=posall[0:4563,]
    #postest=posall[4563:5563]
    y_pos=np.ones([6289,1])
    #y_test_pos=np.ones([1000,1])
    #y_train_pos=np.squeeze(y_train_pos)
    #y_test_pos=np.squeeze(y_test_pos)

    negall=np.array(neg_all)
    #negtrain=negall[0:4563,]
    #negtest=negall[4563:5563]
    y_neg=np.zeros([6289,1])
    data = np.vstack([posall,negall])
    data_w2v = pd.read_table("../../all_nsv_w2v.txt",sep='\s+',header=None)
    data_w2v = np.array(data_w2v)
    data = np.concatenate([data,data_w2v],axis=1)
    y = np.vstack([y_pos,y_neg])
    y = np.squeeze(y)

    #X_train=np.vstack([postrain,negtrain])
    #X_test=np.vstack([postest,negtest])
    #y_train=np.vstack([y_train_pos,y_train_neg])
    #y_test=np.vstack([y_test_pos,y_test_neg])
    #y_train=np.squeeze(y_train)
    #y_test=np.squeeze(y_test)
    train_idx = list(range(5289)) + list(range(6289,11578))
    test_idx = list(range(5289,6289)) + list(range(11578,12578))
    X_train = data[train_idx,]
    X_test = data[test_idx,]
    y_train = y[train_idx]
    y_test = y[test_idx]

    # 配置.fasta文件路径
    pos_seqs = readfa('../../../fasttext/AT6289_pos_new1.fasta')
    neg_seqs = readfa('../../../fasttext/AT6289_neg_new1.fasta')

    #--------------------------- ———————————————————————————————————#

    if contrl == 2:
        all_seqs = pos_seqs + neg_seqs
        all_seqs = np.array(all_seqs)
        seq_train = all_seqs[train_idx]
        seq_test = all_seqs[test_idx]
        print("读取数据完成...")
        return X_train, X_test, y_train, y_test, seq_train, seq_test

    print("读取数据完成...")
    # X_train, X_test, y_train, y_test = train_test_split(data, y, stratify=y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test
