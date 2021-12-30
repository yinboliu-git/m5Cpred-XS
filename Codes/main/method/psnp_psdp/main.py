# !/usr/bin/python3
# -*- codeing = utf-8 -*-
# @Time : 4/25/2021 8:48 PM
# @Author : Liu
# @File : main.py
# @Software : PyCharm

import scipy.io as sio
import time
# import PSDP
# import PSDP_FOR
# import old.PSDP as ps
# import PSNP
import PSNP.PSNP as psn
import PSNP.PSNP_VC as pvc  # 这里可能会报错，请暂时忽略


# 导入matlab数据
mat_filename = u'mp.mat'
data = sio.loadmat(mat_filename)

########### 数据整理 #####################
trianData = data['data1']
testData = data['data2']
trainLabel = data['label'][:,0]

# trianData = pd.Series(data['data1'])
# testData = pd.Series(data['data2'])
# trainLabel = pd.Series(data['label'][:,0])
# tl=trainLabel.loc[1:98].append(trainLabel.loc[9400:9401])
# td = trianData.loc[1:98].append(trianData.loc[9400:9401])
# te = testData.loc[1:100]
# print(td)
# a,b=PSNP.PSNP(td.tolist(),te.tolist(),tl.tolist())
#################################################
start_time=time.time()  # 开始时间  标签：测试时间

############ 算法应用 #################
a,b = psn.PSNP(trianData,testData,trainLabel)
# a,b = psn.PSNP(trianData,testData,trainLabel)
# a,b=PSDP.PSDP(td.tolist(),te.tolist(),tl.tolist(), 3)
# a,b = PSDP_FOR.PSDP(td,te,tl,3)
# a,b = ps.PSDP(trianData,testData,trainLabel,3)
#####################################

end_time=time.time()   # 结束时间  标签：测试时间
print("time:%f" % (end_time-start_time))
# print(a)
# print(b)



