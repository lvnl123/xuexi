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
z=ber_nb.predict(np.c_[xx.ravel(),yy.ravel()]).reshape(xx.shape)
plt.pcolormesh(xx,yy,z,cmap=plt.cm.hot)
plt.scatter(x_train[:,0],x_train[:,1],c=y_train,cmap=plt.cm.Paired,edgecolors='k',label='训练数据')
plt.scatter(x_test[:,0],x_test[:,1],c=y_test,cmap=plt.cm.Paired,edgecolors='k',label='测试数据')
plt.xlim(xx.min(),xx.max())
plt.ylim(yy.min(),yy.max())
plt.xlabel('特征1')
plt.ylabel('特征2')
plt.title('分类结果展示')
plt.legend()
plt.show()

from sklearn.naive_bayes import GaussianNB
gau_nb=GaussianNB()
gau_nb.fit(x_train,y_train)
print("高斯朴素贝叶斯模型得分：{:.3f}".format(gau_nb.score(x_train,y_train)))
z=gau_nb.predict(np.c_[xx.ravel(),yy.ravel()]).reshape(xx.shape)
plt.pcolormesh(xx,yy,z,cmap=plt.cm.Pastel1)
plt.scatter(x_train[:,0],x_train[:,1],c=y_train,cmap=plt.cm.Paired,edgecolors='k',label='train')
plt.scatter(x_test[:,0],x_test[:,1],c=y_test,cmap=plt.cm.Paired,edgecolors='k',label='test')
plt.xlim(xx.min(),xx.max())
plt.ylim(yy.min(),yy.max())
plt.xlabel('feature1')
plt.ylabel('feature2')
plt.title('Classifier: Gaussian')
plt.legend()
plt.show()

