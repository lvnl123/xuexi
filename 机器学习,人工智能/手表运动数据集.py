import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.naive_bayes import BernoulliNB,GaussianNB
from sklearn.ensemble import RandomForestClassifier

data = pd.read_csv('dailyActivity_merged.csv')
#print(data.head())

X= data[['TotalSteps','TotalDistance','TrackerDistance','VeryActiveDistance'
    ,'ModeratelyActiveDistance','LightActiveDistance','VeryActiveMinutes',
         'FairlyActiveMinutes','LightlyActiveMinutes']]

X.corr()
X = X.to_numpy()
y = data[['Calories']]
y= y.to_numpy().ravel()

min_max_scaler = MinMaxScaler()
min_max_scaler.fit(X)
x = min_max_scaler.transform(X)

X_train,X_test,y_train,y_test =  train_test_split(X,y,test_size=0.2,random_state=0,shuffle=True)
print(X_train.shape,y_train.shape,X_test.shape,y_test.shape)

#模型训练
lr = LinearRegression()
lr.fit(X_train,y_train)
bnb = BernoulliNB()
bnb.fit(X_train,y_train)
gnb = GaussianNB()
gnb.fit(X_train,y_train)
rfr = RandomForestClassifier()
rfr.fit(X_train,y_train)
print('线性模型：{}，伯努利朴素贝叶斯：{}，高斯朴素贝叶斯：{}，随机森林回归：{}'.format(lr.score(X_test,y_test),bnb.score(X_test,y_test),gnb.score(X_test,y_test),rfr.score(X_test,y_test)))

from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
for i in range(1,9):
    pca = PCA(n_components=i)
    X_pca = pca.fit_transform(X)
    X_train,X_test,y_train,y_test =  train_test_split(X_pca,y,test_size=0.2,random_state=0,shuffle=True)
    rfr = RandomForestRegressor()
    rfr.fit(X_train, y_train)

    print('i:{},score:{}'.format(i,rfr.score(X_test,y_test)))

from sklearn.model_selection import GridSearchCV
pca = PCA(n_components=9) #值为8时的效果最好
X_pca = pca.fit_transform(X)
X_train,X_test,y_train,y_test =  train_test_split(X_pca,y,test_size=0.2,random_state=0,shuffle=True)
rfr = RandomForestRegressor(n_jobs=-1)
cv = GridSearchCV(rfr, param_grid={'n_estimators': [100,90,80,70,60,50,40,30,20,10,8,5,4],
                                   'max_depth': [30,20, 10, 5, 3],
                                   'min_samples_split': [2, 3, 4, 5],
                                   'min_samples_leaf': [1, 2, 3]})

cv.fit(X_train, y_train)
print('最优秀的模型：',cv.best_params_,'最优分数：',cv.best_score_)
print(cv.best_estimator_.score(X_test,y_test))