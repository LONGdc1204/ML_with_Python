import numpy as np 
import matplotlib.pyplot as plt 
from sklearn import neighbors, datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

iris = datasets.load_iris()
iris_X = iris.data
iris_y = iris.target
print('Number of classes: %d' %len(np.unique(iris_y)))
print('Number of data points: %d' %len(iris_y))

X_train, X_test, y_train, y_test = train_test_split(iris_X, iris_y, test_size = 50)
print('Training size: %d' %len(y_train))
print('Test size    : %d' %len(y_test))

k = int(input('K = '))

clf = neighbors.KNeighborsClassifier(n_neighbors = k, p = 2)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

print ("Print results for 20 test data points:")
print ("Predicted labels: ", y_pred[20:40])
print ("Ground truth    : ", y_test[20:40])

print ("Accuracy of %dNN: %.2f %%" %(k, (100*accuracy_score(y_test, y_pred))))