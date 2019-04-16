import os
import sys
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn import preprocessing
from sklearn.externals import joblib
from sklearn import metrics

import seaborn
import matplotlib.pyplot as plt
import tensorflow as tf

# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
def stop():
    while 1:
        pass
        
        
        
        


# Multinomial Naive Bayes Classifier  
def naive_bayes_classifier(train_x, train_y):  
    from sklearn.naive_bayes import MultinomialNB  
    model = MultinomialNB(alpha=0.01)  
    model.fit(train_x, train_y)  
    return model  
  
  
# KNN Classifier  
def knn_classifier(train_x, train_y):  
    from sklearn.neighbors import KNeighborsClassifier  
    model = KNeighborsClassifier()  
    model.fit(train_x, train_y)  
    return model  
  
  
# Logistic Regression Classifier  
def logistic_regression_classifier(train_x, train_y):  
    from sklearn.linear_model import LogisticRegression  
    model = LogisticRegression(penalty='l2')  
    model.fit(train_x, train_y)  
    return model  
  
  
# Random Forest Classifier  
def random_forest_classifier(train_x, train_y):  
    from sklearn.ensemble import RandomForestClassifier  
    model = RandomForestClassifier(n_estimators=8)  
    model.fit(train_x, train_y)  
    return model  
  
  
# Decision Tree Classifier  
def decision_tree_classifier(train_x, train_y):  
    from sklearn import tree  
    model = tree.DecisionTreeClassifier()  
    model.fit(train_x, train_y)  
    return model  
  
  
# GBDT(Gradient Boosting Decision Tree) Classifier  
def gradient_boosting_classifier(train_x, train_y):  
    from sklearn.ensemble import GradientBoostingClassifier  
    model = GradientBoostingClassifier(n_estimators=200)  
    model.fit(train_x, train_y)  
    return model  
  
  
# SVM Classifier  
def svm_classifier(train_x, train_y):  
    from sklearn.svm import SVC  
    model = SVC(kernel='rbf', probability=True)  
    model.fit(train_x, train_y)  
    return model
      

# SVM Classifier using cross validation  
def svm_cross_validation(train_x, train_y):  
    from sklearn.model_selection import GridSearchCV  
    from sklearn.svm import SVC  
    model = SVC(kernel='rbf', probability=True)  
    param_grid = { 'gamma': [0.2] } #'C': [22],
    grid_search = GridSearchCV(model, param_grid, n_jobs = -1, verbose=1)  
    grid_search.fit(train_x, train_y)  
    best_parameters = grid_search.best_estimator_.get_params()  
    for para, val in list(best_parameters.items()):  
        print(para, val)  
    model = SVC(kernel='rbf', C=22, gamma=best_parameters['gamma'], probability=True)  #best_parameters['C']
    model.fit(train_x, train_y)  
    return model 

        
        

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



clf = svm_cross_validation(train_data,labels_train_src)

joblib.dump(clf, "./sk-module/train_model")

res = clf.predict(test_data)

print("score :",((labels_test_src.values - res)**2).sum()/res.shape[0])




# print(train_data)
# print(test_data)


stop()

ax = []                    # 定义一个 x 轴的空列表用来接收动态的数据
ay = []                    # 定义一个 y 轴的空列表用来接收动态的数据
ay2 = []


TRAIN_DATA_SIZZE = 25000
BATCH_SIZE = 1000

LEARNING_RATE_BASE = 0.9
LEARNING_RATE_DECAY = 0.99
REGULARIZATION_RATE = 0.069
MOVING_AVERAGE_DECAY = 0.99

MODEL_SAVE_PATH="modules/"
MODEL_NAME="mnist_model"
TRAINING_STEPS = 9000000000

input_num = 100
layer_node_num = [400, 200, 100]
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
train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)

with tf.control_dependencies([train_step, variables_averages_op]):
    train_op = tf.no_op(name='train')

saver = tf.train.Saver()

with tf.Session() as sess:
    tf.global_variables_initializer().run()
    
    plt.ion()                  # 开启一个画图的窗口
    for step_count in range(TRAINING_STEPS):
        xs = get_next_batch(train_data,BATCH_SIZE,step_count)
        ys = get_next_batch(train_labels,BATCH_SIZE,step_count)
        
        xt = test_data
        yt = test_labels
        
        _, loss_value, step ,np_output_labels= sess.run([train_op, loss, global_step, y], feed_dict={x: xs, y_: ys})
        
        

        
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
        print("loss :",loss_value)
        print("训练集:\n","\t","错误率：",err_sum/result.shape[0],"@@@@@  正确率：",right_sum/result.shape[0])
        
        
        ay2.append(right_sum/result.shape[0])            # 添加 i 的平方到 y 轴的数据中
        
        
        output_labels_size,_ = np_test_output_labels.shape   #计算测试集的错误率
        result = np.zeros((output_labels_size))
        
        # print(np.argmax(np_test_output_labels))
        result_test = np.zeros((output_labels_size))
        
        for i in range(output_labels_size):
            maxarg = np.argmax(np_test_output_labels[i])
            result_test[i] = maxarg
            np_test_output_labels[i] = 0
            np_test_output_labels[i][maxarg] = 1
            result[i] = np.logical_not(np.logical_not(np_test_output_labels[i] == yt[i]).any())
            pass
        

        print("score :",((labels_test_src.values - result_test)**2).sum()/result_test.shape[0])
        right_sum = result.sum()
        err_sum = np.logical_not(result).sum()
        
        print("测试集：\n","\t","错误率：",err_sum/result.shape[0],"@@@@@  正确率：",right_sum/result.shape[0])
        
        ax.append(step_count)               # 添加 i 到 x 轴的数据中
        ay.append(right_sum/result.shape[0])            # 添加 i 的平方到 y 轴的数据中
        plt.clf()                  # 清除之前画的图
        plt.plot(ax,ay,color = 'red')            # 画出当前 ax 列表和 ay 列表中的值的图形
        plt.plot(ax,ay2)            # 画出当前 ax 列表和 ay 列表中的值的图形
        plt.pause(0.01)             # 暂停一秒
        
        saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME))
    plt.ioff()                 # 关闭画图的窗口
    
    



# print(base_labelsset)
# base_dataset.loc[:,'happiness'].plot()
# plt.show()