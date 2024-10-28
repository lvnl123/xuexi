import numpy as np

from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
wine = load_wine()
X = wine['data']
y = wine['target']
print(X,y)

X_train ,X_test,y_train,y_test = train_test_split(X,y,test_size= 0.2)
print(X_train.shape,X_test.shape,y_train.shape,y_test.shape)

dtc = DecisionTreeClassifier()
rfc = RandomForestClassifier()

dtc.fit(X_train,y_train)
rfc.fit(X_train,y_train)

print('决策树分类：{}，随机森林分布{}'.format(dtc.score(X_test,y_test),rfc.score(X_test,y_test)))

from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
rfc_l = []
dtc_l = []

for i in range(10):
    dtc = DecisionTreeClassifier()
    rfc = RandomForestClassifier()
    dtc_s = cross_val_score(dtc,X,y,cv = 3).mean()
    rfc_s = cross_val_score(rfc,X, y,cv = 3).mean()
    dtc_l.append(dtc_s)
    rfc_l.append(rfc_s)

print(rfc_l)
print(dtc_l)

plt.figure(figsize=(10,5))
plt.plot(range(len(dtc_l)),dtc_l,'r-',label = 'DecisionTreeClassifier')
plt.plot(range(len(rfc_l)),rfc_l,'b--',label = 'RandomForestClassifier')
plt.legend()
plt.show()