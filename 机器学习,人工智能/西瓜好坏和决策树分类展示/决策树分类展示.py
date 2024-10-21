import graphviz
from sklearn.tree import export_graphviz
dtc1=DecisionTreeClassifier(criterion='entropy',max_depth=3,random_state=30)
dtc1.fit(X_train,y_train)
dot_data = export_graphviz(dtc1,impurity=False)
graph = graphviz.Source(dot_data)
graph.render('melon_tree1')
graph

dtc2=DecisionTreeClassifier(criterion='gini',random_state=30)
dtc2.fit(X_train,y_train)
dot_data = export_graphviz(dtc2,impurity=False)
graph = graphviz.Source(dot_data)
graph.render('melon_tree2')
graph