from sklearn import tree

#Apples/Oranges Classifier
#weight and smoothness
features = [[140,1],[130,1],[150,0],[170,0]]
labels = [0,0,1,1]

#Train classifier with decision tree
clf = tree.DecisionTreeClassifier()
clf = clf.fit(features,labels)

#Predict
print clf.predict([[150,1]])
