from sklearn import datasets
iris=datasets.load_iris()

from sklearn.model_selection import train_test_split
feature=iris.data
label=iris.target
x_train,x_test,y_train,y_test=train_test_split(feature,label,test_size=0.25,random_state=62)
from sklearn import svm
svm_classifier=svm.SVC(C=1.0,kernel='rbf',decision_function_shape='ovr',gamma='auto')
svm_classifier.fit(x_train,y_train)
print('训练集:',svm_classifier.score(x_train,y_train))
print('测试集:',svm_classifier.score(x_test,y_test))

