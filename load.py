import tensorflow as tf
import numpy as np
import pandas as pd

data = pd.read_csv("./src-data/happiness_test_abbr.csv")

pd_src_id = data['id']

data.drop(['survey_time'],axis=1,inplace=True)         #去除年月日
data.drop(['id'],axis=1,inplace=True)         #去除id
data = data.dropna(axis=1, how='any')                  #去除有缺省列

data.loc[data['income']>1200000,'income'] = 1200000                #工资太高的给他限制在范围内
data.loc[data['family_income']>2500000,'family_income'] = 2500000  #工资太高的给他限制在范围内

data = data.reset_index()                  #重设索引
data.drop(['index'],axis=1,inplace=True)   #去除多余索引

data_proced = data.loc[:] - data.mean()                                                     #去均值
data_proced = (data_proced - data_proced.min()) / (data_proced.max() - data_proced.min())   #数据归一化


# print(data_proced)
# print(pd_src_id)

np_data = data_proced.values
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
    pd_temp.to_csv("./temp-data/result.csv",index=False)
    # print(result)
