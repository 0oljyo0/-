import os
import sys
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn import preprocessing
from sklearn.externals import joblib

import seaborn
import matplotlib.pyplot as plt
import tensorflow as tf

# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
def stop():
    while 1:
        pass

def data_classify(data_src, label_name, data_train_proportion):    #把乱的数据根据标签分好类 变成有序数据 再分成训练集和测试集两部分 分为两个list返回 data_src存放从文件读来得原始数据 label_name存放的是用于分类的列的名字 data_train_proportion是训练集占总的比列
    labels_class_num = data_src.loc[:,label_name].nunique()
    labels_classes = data_src.loc[:,label_name].unique()
    
    order=np.argsort(labels_classes)
    labels_classes = labels_classes[order]
    
    temp_train = []
    temp_test = []
    
    for lab in labels_classes:
        classfy_pos = data_src.loc[data_src[label_name] == lab,:].index.values
        data_classed = data_src.loc[classfy_pos,:]
        
        data_classed = data_classed.reset_index()                  #重设索引
        data_classed.drop(['index'],axis=1,inplace=True)   #去除多余索引
        
        row_num,col_num = data_classed.shape
        train_row_num = row_num * data_train_proportion
        
        train_data = data_classed.loc[0:(train_row_num)]
        test_data = data_classed.loc[train_row_num:]
        
        train_data = train_data.reset_index()                  #重设索引
        train_data.drop(['index'],axis=1,inplace=True)   #去除多余索引
        
        test_data = test_data.reset_index()                  #重设索引
        test_data.drop(['index'],axis=1,inplace=True)   #去除多余索引
        
        temp_train.append(train_data)
        temp_test.append(test_data)
    return temp_train,temp_test        
        
def where_is_nan(data):    #查数据中缺省的位置  pd的格式
    temp = 0
    print(sys._getframe().f_back.f_lineno)
    for columname in data:
        if(np.isnan(data.loc[:,columname]).any()):
            temp = 1
            print(columname)
            print("\tsum = ",np.isnan(data.loc[:,columname]).sum(),"\n\tpos = ",np.where(np.isnan(data.loc[:,columname]) == True)[0])
    if temp == 0 :
        print("have no nan\n")

def drop_index_rule(data_src,rule):
    drop_pos = data_src.loc[rule,:].index.values
    data_src.drop(drop_pos,axis=0,inplace=True)         #去除年月日
    
def output_all_col_dtype(data_src):
    for col in data_src:
        print(data_src[col].dtype)
        
def data_expand(data_list,row_expand_aim_num):                   #数据扩增 解决样本不平衡 待扩增的数据以list的新式放进去 row_expand_aim_num是想要扩增为多少行
    # data_list[0] = pd.concat( [data_list[0], data_list[0]], axis = 0)
    for count in range(len(data_list)):
        row_num = data_list[count].shape[0]
        if row_num < row_expand_aim_num:
            err = row_expand_aim_num - row_num
            temp = data_list[count].sample(n = err, replace=True)
            data_list[count] = pd.concat( [data_list[count], temp], axis = 0)
        data_list[count] = data_list[count].reset_index()                  #重设索引
        data_list[count].drop(['index'],axis=1,inplace=True)   #去除多余索引

def pack_data_list(data_list):
    temp = data_list[0]
    row_num = len(data_list)
    for count in range(1,row_num):
        temp = pd.concat( [temp, data_list[count]], axis = 0)
    temp = temp.reset_index()                  #重设索引
    temp.drop(['index'],axis=1,inplace=True)   #去除多余索引
    return temp
    
def get_next_batch(all_data,batch_size,step):
    row_num = all_data.shape[0]
    batch_num = row_num/batch_size
    batch_count = step%batch_num
    begin = int(batch_count * batch_size)
    end = int((batch_count + 1) * batch_size)
    return all_data[begin:end]
    
base_dataset = pd.read_csv("./src-data/happiness_train_complete.csv",encoding='ISO-8859-1')


drop_col = ['invest_3','invest_2','invest_0','property_7','property_6','property_5','property_4','property_3','property_0','s_income','edu_status','social_friend','survey_time','invest_other','edu_other','property_other','edu_yr','join_party','minor_child','marital_1st','s_birth','marital_now','s_edu','s_political','s_hukou','s_work_exper','s_work_status','s_work_type','social_neighbor','work_status','work_yr','work_type','work_manage','invest_4','invest_5','invest_6','invest_7','invest_8']
base_dataset.drop(drop_col,axis=1,inplace=True)         #去除缺省且不必要的列

drop_index_rule(base_dataset,np.isnan(base_dataset.hukou_loc))     #去除缺省行
drop_index_rule(base_dataset,np.isnan(base_dataset.family_income))
drop_index_rule(base_dataset,base_dataset.happiness<1) 


base_dataset = base_dataset.reset_index()                  #重设索引
base_dataset.drop(['index','id'],axis=1,inplace=True)   #去除多余索引

# base_labelsset = base_dataset['happiness']
# base_dataset.drop(['happiness'],axis=1,inplace=True)         #去除缺省且不必要的列
train_data, test_data = data_classify(base_dataset, 'happiness', 0.8) 

data_expand(train_data, 5000)                                           #由于样本严重不平衡 这里先进行样本扩增 每一个happiness类都扩到2500个样本 用复制的方法扩增
data_expand(test_data, 1000)

train_data = pack_data_list(train_data)                                 #把数据分成了几类的list给打包到一起
test_data = pack_data_list(test_data)



train_data = train_data.sample(frac=1).reset_index(drop=True)                   #打乱样本顺序
test_data = test_data.sample(frac=1).reset_index(drop=True)
train_data = train_data.sample(frac=1).reset_index(drop=True)                   #打乱样本顺序
test_data = test_data.sample(frac=1).reset_index(drop=True)
train_data = train_data.sample(frac=1).reset_index(drop=True)                   #打乱样本顺序
test_data = test_data.sample(frac=1).reset_index(drop=True)
train_data = train_data.sample(frac=1).reset_index(drop=True)                   #打乱样本顺序
test_data = test_data.sample(frac=1).reset_index(drop=True)

where_is_nan(train_data)
where_is_nan(test_data)

labels_train = train_data['happiness']
labels_test = test_data['happiness']

labels_test_src = labels_test
labels_train_src = labels_train

train_data.drop(['happiness'],axis=1,inplace=True)         #去除缺省且不必要的列
test_data.drop(['happiness'],axis=1,inplace=True)         #去除缺省且不必要的列


np_src_happiness = labels_train.values                             #获取原始的幸福感数据 numpy格式                 #训练labels
np_happiness_train_labels = np.zeros((np_src_happiness.shape[0],5),dtype = np.int)  #创建可以用于训练的np 标签 5分类问题
for i in range(np_src_happiness.shape[0]):
    np_happiness_train_labels[i,np_src_happiness[i]-1] = 1

np_src_happiness = labels_test.values                             #获取原始的幸福感数据 numpy格式                 #测试labels
np_happiness_test_labels = np.zeros((np_src_happiness.shape[0],5),dtype = np.int)  #创建可以用于训练的np 标签 5分类问题
for i in range(np_src_happiness.shape[0]):
    np_happiness_test_labels[i,np_src_happiness[i]-1] = 1

train_data = preprocessing.scale(train_data)   #数据标准化 使数据的每列基本满足正态分布
test_data = preprocessing.scale(test_data)

train_labels = np_happiness_train_labels   
test_labels = np_happiness_test_labels


print(train_data.shape,test_data.shape)
print(train_labels.shape,test_labels.shape)

clf = joblib.load("./sk-module/train_model")

res = clf.predict(test_data)

print("score :",((labels_test_src.values - res)**2).sum()/res.shape[0])




# print(train_data)
# print(test_data)


stop()