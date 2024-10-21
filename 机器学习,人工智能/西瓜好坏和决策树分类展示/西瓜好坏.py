import pandas as pd
data=pd.read_csv('melon_data.csv', encoding='utf-8')
print('西瓜的数据维度:',data.shape)
print('前十条数据:',data.head(10))


data.loc[data['好瓜']!='是','好瓜']=0
data.loc[data['好瓜']=='是','好瓜']=1
data['好瓜']=data['好瓜'].astype('int')
print('修改目标值之后的数据集',data.head(10))

data_x=pd.get_dummies(data.iloc[:,1:-1])
print('转换后的特征值',data_x.head())
X=data_x.values
y=data.iloc[:,-1].values
print('特征维度{}标签维度{}'.format(X.shape,y.shape))

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.8,random_state=125)
dtc=DecisionTreeClassifier(criterion='entropy',max_depth=3,random_state=30)
dtc.fit(X_train,y_train)
print('训练集得分:{:.2f}'.format(dtc.score(X_test,y_test)))

y_pred=dtc.predict(X_test)
print('模型预测的分类结果为',y_pred)
print('真实分类为,',y_test)
