import pandas as pd
df = pd.read_csv('adult.csv',header=None,index_col=False,
                 names=['年龄','单位性质','权重','学历','受教育时长','婚姻情况','职业','家庭情况','种族','性别','资产所得','资产损失','周工作时长','原籍','收入'])
print('adult文件的数据形态:',df.shape)
print('输出数据的前5行')
display(df.head())

group_income=df.groupby(by='收入')
income_lessthan50k=dict([x for x in group_income])[' <=50k']
income_morethan50k=dict([x for x in group_income])[' >50k']
print('收入<=50k的样本数量:',income_morethan50k.shape[0])
print('收入>50k的样本数量:',income_morethan50k.shape[0])

data=pd.concat([income_lessthan50k[:10000],income_morethan50k],axis=0)
data=data.sort_index()
print('数据集形态:',data.shape)
print('输出数据集的前10行:')
display(data[:10])

from slearn.preprocessing import LabelEncoder
label_encoder=LabelEncoder()
input_classes=['audi','ford','audi','toyota','ford','bmw']
label_encoder.fit(input_classes)
print('class mapping:')
for i,item in enumerate(label_encoder.classes_):
    print(item,'-->',i)

import  numpy as np
from sklearn.preprocessing import LabelEncoder
def get_data_encoded(data):
    data=np.array(data.astype(str))
    encoder_list=[]
    data_encoded=np.empty(data.shape)
    for i,item in enumerate(data[0]):
        if item.isdigit():
            data_encoded[:,i]=data[:,i]
        else:
            encoder_list.append(LabelEncoder())
            data_encoded[:,i]=encoder_list[-1].fit_transform(data[:,i])
    return data_encoded,encoder_list
data_encoded,encoder_list=get_data_encoded(data)
x=data_encoded[:,:-1].astype(int)
y=data_encoded[:,-1].astype(int)
print('编码处理完成的数据集')
print('特征维度:{},标签维度:{]'.format(x.shape,y.shape))

from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=0)
scaler=StandardScaler().fit(x_train)
x_train_scaled=scaler.transform(x_train)
x_test_scaled=scaler.transform(x_test)
gnb=GaussianNB()
gnb.fit(x_train_scaled,y_train)
print('训练集得分:{:.3f}'.format(gnb.score(x_train_scaled,y_train)))
print('测试集得分:{:.3f}'.format(gnb.score(x_test_scaled,y_test)))

test=income_lessthan50k[10000:10003]
print('选取测试样本:')
display(test)

data_all=pd.concat([data,test])
data_all_encoded,encoder_list=get_data_encoded(data_all)
test_encoded=data_all_encoded[-3:].astype(int)
print('打印编码转换后的测试数据: \n',test_encoded)

test_encoded_x=test_encoded[:,:-1]
test_encoded_y=test_encoded[:,-1]
print('测试样本的特征数据x: \n',test_encoded_x)
print('\n测试样本的收入等级:',test_encoded_y)

test_encoded_x_scaled=scaler.transform(test_encoded_x)
pred_encoded_y=gnb.predict(test_encoded_x_scaled)
print('测试样本的预测分类为: \n',pred_encoded_y)
pred_y=[encoder_list[-1].inverse_transform(pred_encoded_y)
print('预测收入等级:',pred_y)