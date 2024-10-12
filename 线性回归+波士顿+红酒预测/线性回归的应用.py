# import numpy as np
# import matplotlib.pyplot as plt
#
# from sklearn.datasets import make_regression
# from sklearn.linear_model import LinearRegression
# from sklearn.metrics import mean_squared_error
# x,y=make_regression(n_samples=50,n_features=1,noise=50,random_state=3,n_informative=1)
#
# lr=LinearRegression()
# lr.fit(x,y)
#
# z=np.linspace(-3,3,200).reshape(-1,1)
# plt.scatter(x,y,c='b',s=60)
# plt.plot(z,lr.predict(z),c='purple')
#
# plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
# plt.rcParams['axes.unicode_minus']=False #用来正常显示负号
# plt.title('线性回归')
# plt.show()
# print('代码运行结果: \n')
# print('===================================')
# print('直线的斜率是:{:.2f}'.format(lr.coef_[0]))
# print('直线的截距是:{:.2f}'.format(lr.intercept_))
# print('直线方程为:y={:.2f}'.format(lr.coef_[0]),'x','+{:.2f}'.format(lr.intercept_))
# print('===================================')


import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

# 生成数据
x,y=make_regression(n_samples=100,n_features=2,random_state=80,n_informative=2)

# 创建并训练模型
lr=LinearRegression()
lr.fit(x,y)

# 生成测试数据
z=np.linspace(-3,3,200).reshape(-1,2)

# 分别绘制两个特征的散点图
plt.scatter(x[:, 0], y, c='b', s=60)
plt.scatter(x[:, 1], y, c='r', s=60)

# 绘制回归线
plt.plot(z[:, 0], lr.predict(z), c='purple')
plt.plot(z[:, 1], lr.predict(z), c='purple')

# 设置中文标签
plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号

# 设置标题
plt.title('线性回归')

# 显示图形
plt.show()

# 划分数据集
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=8)

# 创建并训练模型
lr2=LinearRegression()
lr2.fit(x_train, y_train)

# 预测
y_predict = lr2.predict(x_test)

# 计算预测误差
mse = mean_squared_error(y_test, y_predict)

print('')
print('==========================')
print('lr2.coef_:{}'.format(lr2.coef_))
print('lr2.intercept_:{}'.format(lr2.intercept_))
print('预测误差(MSE):{}'.format(mse))
print('==========================')


