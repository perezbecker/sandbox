# import a dataset

from sklearn import datasets
iris = datasets.load_iris()

X=iris.data
Y=iris.target

from sklearn.cross_validation import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.5)

#tree classifier
from sklearn import tree
tree_classifier = tree.DecisionTreeClassifier()
tree_classifier.fit(X_train,Y_train)

#KNeighborsClassifier
from sklearn.neighbors import KNeighborsClassifier
kn_classifier = KNeighborsClassifier()
kn_classifier.fit(X_train,Y_train)


tree_predictions = tree_classifier.predict(X_test)
kn_predictions = kn_classifier.predict(X_test)


from sklearn.metrics import accuracy_score
print accuracy_score(Y_test,tree_predictions)
print accuracy_score(Y_test,kn_predictions)
