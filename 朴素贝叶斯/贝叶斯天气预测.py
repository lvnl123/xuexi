import numpy as np
x=np.array([[0,1,0,1],
           [1,1,1,0],
           [0,1,1,0],
           [0,0,0,1],
           [0,1,1,0],
           [0,1,0,1],
           [1,0,0,1]])
y=np.array([0,1,1,0,1,0,0])
counts={}
for label in np.unique(y):
    counts[label]=x[y==label].sum(axis=0)
print('feature counts:\n{}'.format(counts))

from sklearn.naive_bayes import BernoulliNB
clf=BernoulliNB()
clf.fit(x,y)
next_day1=[[0,0,1,0]]
pred_day1=clf.predict(next_day1)
if pred_day1[0]==[1]:
    print('预测这一天会下雨')
else:
    print('预测这一天不会下雨')

pred_prob1=clf.predict_proba(next_day1)
print('预测这一天会下雨的概率：{0:.2f},预测这一天不会下雨的概率为{1:.2f}'.format(pred_prob1[0,0],pred_prob1[0,1]))

next_day2=[[1,1,0,1]]
pred_day2=clf.predict(next_day2)
if pred_day2[0]==1:
    print('预测另一天会下雨')
else:
    print('预测另一天不会下雨')
pred_prob2=clf.predict_proba(next_day2)
print('预测另一天会下雨的概率为：{0:.2f},预测另一天不会下雨的概率为{1:.2f}'.format(pred_prob2[0,0],pred_prob2[0,1]))

pred_days=[[0,0,1,0],[1,1,0,1]]
print('这两天的预测结果是:',clf.predict(pred_days))
print('这两天的预测概率为: \n',clf.predict_proba(pred_days))