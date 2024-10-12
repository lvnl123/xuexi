from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB
x,y=make_blobs(n_samples=400,centers=4,random_state=0)
x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=8)
ber_nb=BernoulliNB()
ber_nb.fit(x_train,y_train)
print("伯努利朴素贝叶斯模型得分：{:.3f}".format(ber_nb.score(x_train,y_train)))

import matplotlib.pyplot as plt
import numpy as np
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus']=False
x_min,x_max=x[:,0].min()-0.5,x[:,0].max()+0.5
y_min,y_max=x[:,1].min()-0.5,x[:,1].max()+0.5
xx,yy=np.meshgrid(np.arange(x_min,x_max,.02),np.arange(y_min,y_max,.02))

from sklearn.naive_bayes import GaussianNB
gau_nb=GaussianNB()
gau_nb.fit(x_train,y_train)
print("高斯朴素贝叶斯模型得分：{:.3f}".format(gau_nb.score(x_train,y_train)))
z=gau_nb.predict(np.c_[xx.ravel(),yy.ravel()]).reshape(xx.shape)
################################################
#多项贝叶斯
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler()
scaler.fit(x_train)
x_train_scaled=scaler.transform(x_train)
x_test_scaled=scaler.transform(x_test)
mul_nb= MultinomialNB()
mul_nb.fit(x_train_scaled,y_train)
print("多项朴素贝叶斯模型得分：{:.3f}".format(mul_nb.score(x_test_scaled,y_test)))

z_scaled=scaler.transform(np.c_[xx.ravel(),yy.ravel()])
z=mul_nb.predict(z_scaled).reshape(xx.shape)
plt.pcolormesh(xx,yy,z,cmap=plt.cm.Pastel1)
plt.scatter(x_train[:,0],x_train[:,1],c=y_train,cmap=plt.cm.hot,edgecolors='k',label='train')
plt.scatter(x_test[:,0],x_test[:,1],c=y_test,cmap=plt.cm.hot,marker='*',label='test')
plt.xlim(xx.min(),xx.max())
plt.ylim(yy.min(),yy.max())
plt.xlabel('特征1')
plt.ylabel('特征2')
plt.title('MultinomialNB 分类结果展示')
plt.legend()
plt.show()
#离散化贝叶斯
from sklearn.preprocessing import KBinsDiscretizer
kbs=KBinsDiscretizer(n_bins=10,encode='onehot').fit(x_train)
x_train_bins=kbs.transform(x_train)
x_test_bins=kbs.transform(x_test)
mul_nb= MultinomialNB().fit(x_train_bins,y_train)
y_pred=mul_nb.predict(x_test_bins)
print("离散化多项朴素贝叶斯模型得分：{:.3f}".format(mul_nb.score(x_test_bins,y_test)))

z=np.c_[(xx.ravel(),yy.ravel())]
z_bins=kbs.transform(z)
z=mul_nb.predict(z_bins).reshape(xx.shape)
plt.pcolormesh(xx,yy,z,cmap=plt.cm.Pastel1)
plt.scatter(x_train[:,0],x_train[:,1],c=y_train,cmap=plt.cm.hot,edgecolors='k',label='train')
plt.scatter(x_test[:,0],x_test[:,1],c=y_test,cmap=plt.cm.hot,marker='*',label='test')
plt.xlim(xx.min(),xx.max())
plt.ylim(yy.min(),yy.max())
plt.xlabel('feature1')
plt.ylabel('feature2')
plt.title('Classifier: MultinomialNB')
plt.legend()
plt.show()
