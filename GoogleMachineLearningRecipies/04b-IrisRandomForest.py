
import numpy as np
import matplotlib.pyplot as plt

# import a dataset
from sklearn import datasets
iris = datasets.load_iris()

X=iris.data
Y=iris.target

feature_names=iris.feature_names
target_names=iris.target_names


from sklearn.cross_validation import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.5)

#RandomForestClassifier
from sklearn.ensemble import RandomForestClassifier
rf_classifier = RandomForestClassifier(n_estimators=1000,n_jobs= -1)
rf_classifier.fit(X_train,Y_train)
importances = rf_classifier.feature_importances_
std = np.std([tree.feature_importances_ for tree in rf_classifier.estimators_],axis=0)
indices = np.argsort(importances)[::-1]
# Print the feature ranking
print("Feature ranking:")

feature_names_ordered_by_importance=[]
for f in range(X.shape[1]):
    print f+1, feature_names[indices[f]], importances[indices[f]]
    feature_names_ordered_by_importance.append(feature_names[indices[f]])
# Plot the feature importances of the forest
plt.figure()
plt.title("Feature importances")
plt.bar(range(X.shape[1]), importances[indices],
       color="r", yerr=std[indices], align="center")
plt.xticks(range(X.shape[1]), feature_names_ordered_by_importance)
plt.xlim([-1, X.shape[1]])
plt.show()
