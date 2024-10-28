import pandas as pd
import numpy as np
from sklearn.model_selection import  train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

data_url = "http://lib.stat.cmu.edu/datasets/boston"
raw_df = pd.read_csv(data_url, sep="\s+", skiprows=22, header=None)
data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
target = raw_df.values[1::2, 2]

print(data.shape,target.shape)
X_train ,X_test,y_train,y_test = train_test_split(data,target,test_size= 0.2)
print(X_train.shape,X_test.shape,y_train.shape,y_test.shape)

rfr = RandomForestRegressor()
lr = LinearRegression()

rfr.fit(X_train,y_train)
lr.fit(X_train,y_train)


print('随机森林：{}，线性回归：{}'.format(rfr.score(X_test,y_test),lr.score(X_test,y_test)))

plt.figure()
rfr_predict = rfr.predict(X_test)
lr_predict = lr.predict(X_test)

plt.plot(range(len(X_test)),y_test,label = 'original value')
plt.plot(range(len(rfr_predict)),rfr_predict,label = 'RandomForestRegressor')
plt.plot(range(len(lr_predict)),lr_predict,label = 'LinearRegression')
plt.legend()
plt.show()

