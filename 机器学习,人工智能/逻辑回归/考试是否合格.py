import numpy as np
x_train = np.array([(0,0), (2,0.9), (3,0.4), (4,0.9), (5,0.4), (6,0.4), (6, 0.8), (6,0.7),(7,0.2),(7.5, 0.8), (7, 0.9), (8, 0.1), (8, 0.6), (8, 0.8)])
y_train = np.array([0,0,0,1,0,0,1,1,0,1,1,0,1,1])
print('复习情况 x_train: \n', x_train)

from sklearn.linear_model import LogisticRegression
logistic= LogisticRegression(solver='lbfgs',C=10)
logistic.fit(x_train,y_train)
x_test=[(3,0.9),(8,0.5),(7,0.2),(4,0.5),(4,0.7)]
y_test=[0,1,0,0,1]
score=logistic.score(x_test,y_test)
print('模型得分: \n', score)

learning=np.array([(8,0.9)])
result=logistic.predict(learning)
result_proba=logistic.predict_proba(learning)
print('复习时长为:{0},效率为:{1}'.format(learning[0,0],learning[0,1]))
print('不及格概率为:{0:.2f},及格概率为:{1: .2f}'.format(result_proba[0,0],result_proba[0,1]))
print('综合判断期末考试结果:{}'.format('及格'if result==1 else '不及格'))