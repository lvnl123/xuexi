import numpy as np
from sklearn.datasets import load_breast_cancer
cancer=load_breast_cancer()
x=cancer.data
y=cancer.target
print('breast_cancer数据集纬度为: ',x.shape)
print('breast_cancer数据集的类别标签为: ',np.unique(y))
print('肿瘤分类:',cancer['target_names'])

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=23)
print('训练集数据维度:',x_train.shape)
print('测试集数据维度:',y_train.shape)
print('测试集数据维度:',x_test.shape)
print('测试集标签维度:',y_test.shape)

from sklearn.preprocessing import StandardScaler
scaler=StandardScaler().fit(x_train)
x_train_scaled=scaler.transform(x_train)
x_test_scaled=scaler.transform(x_test)
print('标准化前数据集数据的最小值和最大值:{0},{1}'.format(x_train.min(),x_train.max()))
print('标准化后数据集数据的最小值和最大值:{0:.2f},{1:.2f}'.format(x_train_scaled.min(),x_train_scaled.max()))
print('标准化前测试集数据的最小值和最大值:{0},{1}'.format(x_test.min(),x_test.max()))
print('标准化后测试集数据的最小值和最大值:{0:.2f},{1:.2f}'.format(x_test_scaled.min(),x_test_scaled.max()))

from sklearn.linear_model import LogisticRegression
log_reg=LogisticRegression(solver='lbfgs')
log_reg.fit(x_train_scaled,y_train)
print('训练集得分:{:.2f}'.format(log_reg.score(x_train_scaled,y_train)))

print('各特征的相关系数为: \n',log_reg.coef_)
print('模型截距为:',log_reg.intercept_)
print('模型的迭代次数为:',log_reg.n_iter_)

test_score=log_reg.score(x_test_scaled,y_test)
test_pred=log_reg.predict(x_test_scaled)
test_prob=log_reg.predict_proba(x_test_scaled)
print('测试集准确率为:{:.2f}'.format(test_score))
print('预测测试集前5个结果为:',test_pred[:5])
print('测试集前5个对应类别的概率为: \n',np.round(test_prob[:5],2))
