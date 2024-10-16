import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

diabetes=load_diabetes()
diabetes.keys()

x=diabetes['data']
y=diabetes['target']
x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=8)
diabetes_lr=LinearRegression()
diabetes_lr.fit(x_train,y_train)
y_predict=diabetes_lr.predict(x_test)
mse=mean_squared_error(y_test,y_predict)
print('均方误差为：',mse)
print('')
print('=====================')
print('coef_:{}'.format(diabetes_lr.coef_[:]))
print('intercept_:{}'.format(diabetes_lr.intercept_))
print('=====================')

plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_predict, color='blue', alpha=0.5)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=2)
plt.xlabel('实际数值')
plt.ylabel('预测数值')
plt.title('糖尿病数据集的实际值与预测值对比')
plt.show()
