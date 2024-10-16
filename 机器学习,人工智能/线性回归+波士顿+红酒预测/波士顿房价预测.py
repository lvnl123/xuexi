import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

boston = pd.read_csv('BostonHousing.csv')
X = boston.drop('medv', axis=1)
y = boston['medv']

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X, y, random_state=36)
lr_boston = LinearRegression()
lr_boston.fit(x_train, y_train)
y_predict = lr_boston.predict(x_test)

from sklearn.metrics import mean_squared_error
mse = mean_squared_error(y_test, y_predict)
print('线性回归模型的均方误差为: \n', mse)

print('')
print('=================')
print('lr_boston.coef_:{}'.format(lr_boston.coef_))
print('lr_boston.intercept_:{}'.format(lr_boston.intercept_))
print('=================')

plt.figure()
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
x = np.arange(1, y_predict.shape[0] + 1)
plt.plot(x, y_test, marker='o', linestyle=':', markersize=5)
plt.plot(x, y_predict, marker='*', markersize=5)
plt.legend(['真实房价','预测房价'])
plt.title('波士顿真实房价与预测房价走势图')
plt.show()
