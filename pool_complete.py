# -*- coding: utf-8 -*- 
#!/bin/python3
import sys
import os
import csv
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from prettytable import PrettyTable


def stop():
    while 1:
        pass

#np.set_printoptions(threshold=np.inf)
#显示所有列
# pd.set_option('display.max_columns', None)
# 显示所有行
# pd.set_option('display.max_rows', None)
# 设置value的显示长度为100，默认为50
# pd.set_option('max_colwidth',100)

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
    
    
    
data = pd.read_csv("./src-data/happiness_train_complete.csv",encoding='ISO-8859-1')
data_src_row_num,data_src_col_num = data.shape  #获取数据大小

drop_index_rule(data,data.happiness<1)                 #去除有问题的幸福感行

drop_col = ['invest_3','invest_2','invest_0','property_7','property_6','property_5','property_4','property_3','property_0','s_income','edu_status','social_friend','survey_time','invest_other','edu_other','property_other','edu_yr','join_party','minor_child','marital_1st','s_birth','marital_now','s_edu','s_political','s_hukou','s_work_exper','s_work_status','s_work_type','social_neighbor','work_status','work_yr','work_type','work_manage','invest_4','invest_5','invest_6','invest_7','invest_8']
data.drop(drop_col,axis=1,inplace=True)         #去除缺省且不必要的列

drop_index_rule(data,np.isnan(data.hukou_loc))
drop_index_rule(data,np.isnan(data.family_income))

data.loc[data['income']>1200000,'income'] = 1200000                #工资太高的给他限制在范围内
data.loc[data['family_income']>2500000,'family_income'] = 2500000  #工资太高的给他限制在范围内

train_data_classed,test_data_classed = data_classify(data, 'happiness', 0.9)    #把样本按happiness分成几个类 抽80%为训练集 20%为测试集

data_expand(train_data_classed, 5000)                                           #由于样本严重不平衡 这里先进行样本扩增 每一个happiness类都扩到2500个样本 用复制的方法扩增
data_expand(test_data_classed, 1000)

data_train = pack_data_list(train_data_classed)                                 #把数据分成了几类的list给打包到一起
data_test = pack_data_list(test_data_classed)

data_train = data_train.sample(frac=1).reset_index(drop=True)                   #打乱样本顺序
data_test = data_test.sample(frac=1).reset_index(drop=True)
data_train = data_train.sample(frac=1).reset_index(drop=True)                   #打乱样本顺序
data_test = data_test.sample(frac=1).reset_index(drop=True)
data_train = data_train.sample(frac=1).reset_index(drop=True)                   #打乱样本顺序
data_test = data_test.sample(frac=1).reset_index(drop=True)
data_train = data_train.sample(frac=1).reset_index(drop=True)                   #打乱样本顺序
data_test = data_test.sample(frac=1).reset_index(drop=True)

np_src_happiness = data_train['happiness'].values                             #获取原始的幸福感数据 numpy格式                 #训练labels
np_happiness_train_labels = np.zeros((np_src_happiness.shape[0],5),dtype = np.int)  #创建可以用于训练的np 标签 5分类问题
for i in range(np_src_happiness.shape[0]):
    np_happiness_train_labels[i,np_src_happiness[i]-1] = 1

np_src_happiness = data_test['happiness'].values                             #获取原始的幸福感数据 numpy格式                 #测试labels
np_happiness_test_labels = np.zeros((np_src_happiness.shape[0],5),dtype = np.int)  #创建可以用于训练的np 标签 5分类问题
for i in range(np_src_happiness.shape[0]):
    np_happiness_test_labels[i,np_src_happiness[i]-1] = 1
    

data_train.drop('id',axis=1,inplace=True)    
data_train.drop('happiness',axis=1,inplace=True)    

data_test.drop('id',axis=1,inplace=True)    
data_test.drop('happiness',axis=1,inplace=True)    

data_train_proced = data_train.loc[:] - data_train.mean()                                                     #去均值
data_train_proced = (data_train_proced - data_train_proced.min()) / (data_train_proced.max() - data_train_proced.min())   #数据归一化

data_test_proced = data_test.loc[:] - data_test.mean()                                                     #去均值
data_test_proced = (data_test_proced - data_test_proced.min()) / (data_test_proced.max() - data_test_proced.min())   #数据归一化

np_train_data = data_train_proced.values           #准备用于训练的数据和标签 要是array格式
np_train_labels = np_happiness_train_labels

np_test_data = data_test_proced.values            #准备用于测试的数据和标签 要是array格式
np_test_labels = np_happiness_test_labels

where_is_nan(data_train_proced)


where_is_nan(data_test_proced)

print("train data size :", np_train_data.shape)
print("test data size :", np_test_data.shape)



# print(np_train_data[0:6])
# print(np_train_labels[0:6])

# print(get_next_batch(np_train_data,3,1))
# print(get_next_batch(np_train_labels,3,1))
        
# stop()




#以下是构建神经网络部分


ax = []                    # 定义一个 x 轴的空列表用来接收动态的数据
ay = []                    # 定义一个 y 轴的空列表用来接收动态的数据



TRAIN_DATA_SIZZE = 25000
BATCH_SIZE = 1000

LEARNING_RATE_BASE = 0.8
LEARNING_RATE_DECAY = 0.99
REGULARIZATION_RATE = 0.0009
MOVING_AVERAGE_DECAY = 0.99

MODEL_SAVE_PATH="modules/"
MODEL_NAME="mnist_model"
TRAINING_STEPS = 9000000000

input_num = 100
layer_node_num = [300, 200, 100]
output_num = 5

def get_weight_variable(name, shape, regularizer):
    weights = tf.get_variable(name, shape, initializer=tf.truncated_normal_initializer(stddev=0.1))
    if regularizer != None: tf.add_to_collection('losses', regularizer(weights))
    return weights
    
#定义两层简单的网络
x = tf.placeholder(tf.float32, [None, input_num], name='x-input')
y_ = tf.placeholder(tf.float32, [None, output_num], name='y-input')

regularizer = tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)

weights1 = get_weight_variable("weights1",[input_num, layer_node_num[0]], regularizer)
biases1 = tf.get_variable("biases1", [layer_node_num[0]], initializer=tf.constant_initializer(0.0))
layer1 = tf.nn.relu(tf.matmul(x, weights1) + biases1)

weights2 = get_weight_variable("weights2", [layer_node_num[0], layer_node_num[1]],regularizer)
biases2 = tf.get_variable("biases2", [layer_node_num[1]], initializer=tf.constant_initializer(0.0))
layer2 = tf.nn.relu(tf.matmul(layer1, weights2) + biases2)

weights3 = get_weight_variable("weights3", [layer_node_num[1], layer_node_num[2]],regularizer)
biases3 = tf.get_variable("biases3", [layer_node_num[2]], initializer=tf.constant_initializer(0.0))
layer3 = tf.nn.tanh(tf.matmul(layer2, weights3) + biases3)

weights_out = get_weight_variable("weights_out",[layer_node_num[2], output_num], regularizer)
biases_out = tf.get_variable("biases_out", [output_num], initializer=tf.constant_initializer(0.0))
layer_out = tf.matmul(layer3, weights_out) + biases_out
y = layer_out

global_step = tf.Variable(0, trainable=False)

variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
variables_averages_op = variable_averages.apply(tf.trainable_variables())
cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.argmax(y_, 1))
cross_entropy_mean = tf.reduce_mean(cross_entropy)
loss = cross_entropy_mean + tf.add_n(tf.get_collection('losses'))
learning_rate = tf.train.exponential_decay(
    LEARNING_RATE_BASE,
    global_step,
    TRAIN_DATA_SIZZE / BATCH_SIZE, LEARNING_RATE_DECAY,
    staircase=True)
    
# Optimizer 
# GradientDescentOptimizer 
# AdagradOptimizer 
# AdagradDAOptimizer 
# MomentumOptimizer 
# AdamOptimizer 
# FtrlOptimizer 
# RMSPropOptimizer
# train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)
train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss, global_step=global_step)

with tf.control_dependencies([train_step, variables_averages_op]):
    train_op = tf.no_op(name='train')

saver = tf.train.Saver()

with tf.Session() as sess:
    tf.global_variables_initializer().run()
    
    # plt.ion()                  # 开启一个画图的窗口
    for i in range(TRAINING_STEPS):
        xs = get_next_batch(np_train_data,BATCH_SIZE,i)
        ys = get_next_batch(np_train_labels,BATCH_SIZE,i)
        
        xt = np_test_data
        yt = np_test_labels
        
        _, loss_value, step ,np_output_labels= sess.run([train_op, loss, global_step, y], feed_dict={x: xs, y_: ys})
        
        
        # ax.append(i)               # 添加 i 到 x 轴的数据中
        # ay.append(int(loss_value))            # 添加 i 的平方到 y 轴的数据中
        # plt.clf()                  # 清除之前画的图
        # plt.plot(ax,ay)            # 画出当前 ax 列表和 ay 列表中的值的图形
        # plt.pause(0.01)             # 暂停一秒
        
        np_test_output_labels,_ = sess.run([y,loss], feed_dict={x: xt, y_: yt})
        
        output_labels_size,_ = np_output_labels.shape   #计算训练集的错误率
        result = np.zeros((output_labels_size))
        
        for i in range(output_labels_size):
            maxarg = np.argmax(np_output_labels[i])
            np_output_labels[i] = 0
            np_output_labels[i][maxarg] = 1
            result[i] = np.logical_not(np.logical_not(np_output_labels[i] == ys[i]).any())
            #pass
        
        right_sum = result.sum()
        err_sum = np.logical_not(result).sum()
        print(loss_value)
        print("训练集:\n","\t","错误率：",err_sum/result.shape[0],"@@@@@  正确率：",right_sum/result.shape[0])
        
        output_labels_size,_ = np_test_output_labels.shape   #计算测试集的错误率
        result = np.zeros((output_labels_size))
        
        for i in range(output_labels_size):
            maxarg = np.argmax(np_test_output_labels[i])
            np_test_output_labels[i] = 0
            np_test_output_labels[i][maxarg] = 1
            result[i] = np.logical_not(np.logical_not(np_test_output_labels[i] == yt[i]).any())
            pass
        
        right_sum = result.sum()
        err_sum = np.logical_not(result).sum()
        
        print("测试集：\n","\t","错误率：",err_sum/result.shape[0],"@@@@@  正确率：",right_sum/result.shape[0])
        
        saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME))
    # plt.ioff()                 # 关闭画图的窗口
    
    
