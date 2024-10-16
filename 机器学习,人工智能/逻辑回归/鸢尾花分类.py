import numpy as np
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

iris=load_iris()
x=iris.data
y=iris.target

x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=8)
print('训练集数据纬度:', x_train.shape)
print('测试集数据纬度:', x_test.shape)
print('目标分类标签取值为:', np.unique(y))

clf1 = LogisticRegression(solver='liblinear')
clf1.fit(x_train,y_train)
print('训练集得分:{:.2f}'.format(clf1.score(x_train,y_train)))
print('测试集得分:{:.2f}'.format(clf1.score(x_test,y_test)))

clf2 = LogisticRegression(solver='lbfgs')
clf2.fit(x_train,y_train)
print('训练集得分:{:.2f}'.format(clf2.score(x_train,y_train)))
print('测试集得分:{:.2f}'.format(clf2.score(x_test,y_test)))
