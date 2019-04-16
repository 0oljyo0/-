import sys
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn import preprocessing

import seaborn
import matplotlib.pyplot as plt

def stop():
    while 1:
        pass

def drop_index_rule(data_src,rule):
    drop_pos = data_src.loc[rule,:].index.values
    data_src.drop(drop_pos,axis=0,inplace=True)         #去除年月日
    
def drop_nan_columns(data_src,col_name):     #删除某列有nan的行
    drop_pos = data_src.loc[np.isnan(col_name),:].index.values
    data_src.drop(drop_pos,axis=0,inplace=True) 
    # data_src.drop(drop_pos,axis=0,inplace=True)         #去除年月日
    
def get_col_sum(data_src):                            #列出每一列的 负数 和 正数 分别的和
    x= PrettyTable(["col_name", "col_sum +", "col_sum -"])
    for col in data_src:
        x.add_row([col, data_src.loc[data_src.loc[:,col] >= 0,col].sum(), data_src.loc[data_src.loc[:,col] < 0,col].sum()])
    print(x)
    
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
            
def data_classify(data_src, label_name, data_train_proportion):    #把乱的数据根据标签分好类 变成有序数据 再分成训练集和测试集两部分 分为两个list返回 data_src存放从文件读来得原始数据 label_name存放的是用于分类的列的名字 data_train_proportion是训练集占总的比列
    labels_class_num = data_src.loc[:,label_name].nunique()
    labels_classes = data_src.loc[:,label_name].unique()
    
    order=np.argsort(labels_classes)
    labels_classes = labels_classes[order]
    
    temp_train = []
    temp_test = []
    
    for lab in labels_classes:
        classfy_pos = data.loc[data[label_name] == lab,:].index.values
        data_classed = data.loc[classfy_pos,:]
        
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
    

data = pd.read_csv("./src-data/happiness_test_complete.csv",encoding='ISO-8859-1')
colname = pd.read_csv("./temp-data/colname.csv").columns
data_test = data.loc[:,colname]

# data = pd.read_csv("./src-data/happiness_train_complete.csv",encoding='ISO-8859-1')

# data_res = pd.read_csv("./temp-data/result-complete.csv")

# plt.hist(data.happiness, bins=40, normed=0, facecolor="blue", edgecolor="black", alpha=0.7)
# plt.hist(data_res.happiness, bins=40, normed=0, facecolor="red", edgecolor="black", alpha=0.7)

# plt.show()
# stop()

print(data_test)

np_data = preprocessing.scale(data_test)
placehildery = np.zeros((np_data.shape[0],5))

ckpt = tf.train.get_checkpoint_state('./modules/')
saver = tf.train.import_meta_graph(ckpt.model_checkpoint_path +'.meta')

with tf.Session() as sess:
    saver.restore(sess,ckpt.model_checkpoint_path)                        # 载入参数，参数保存在两个文件中，不过restore会自己寻找
    np_output_labels = sess.run(tf.get_default_graph().get_tensor_by_name('add_3:0'),feed_dict={'x-input:0': np_data,'y-input:0':placehildery})
    
    output_labels_size,_ = np_output_labels.shape   #计算训练集的错误率
    for i in range(output_labels_size):
        maxarg = np.argmax(np_output_labels[i])
        np_output_labels[i] = 0
        np_output_labels[i][maxarg] = 1

    np.set_printoptions(threshold=np.inf)
    
    result = np.zeros((np_output_labels.shape[0]))
    for i in range(np_output_labels.shape[0]):
        result[i] = np.argmax(np_output_labels[i]) + 1
    result = result.astype(np.int)
    pd_id = pd.DataFrame(np.array(range(8001,10969)),columns=['id'])
    pd_data = pd.DataFrame(result,columns=['happiness'])
    pd_temp = pd.concat([pd_id,pd_data],axis = 1)
    # pd_data = pd.DataFrame(result,columns=['happiess'])
    print(pd_temp)
    pd_temp.to_csv("./temp-data/result-complete.csv",index=False)
    # print(result)
