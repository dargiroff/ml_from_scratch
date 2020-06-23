from sklearn.datasets import load_iris
import numpy as np
from sklearn.model_selection import train_test_split
from knn import KNN
from matplotlib import pyplot as plt
plt.style.use('bmh')

# Load the data and split it into training and test sets
iris = load_iris()
X, y = iris.data, iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.66, random_state=1337)

# Visualize two of the features
fig, ax = plt.subplots()
ax.scatter(X[:, 0], X[:, 1], s=20, edgecolor='black', c=y)
ax.set_facecolor('white')
plt.show()

# Intialize the classifier
clf = KNN(k=27)
# Fit the model and make a prediction
clf.fit(X_train, y_train)
predictions = clf.predict(X_test)
# Asess the model accuracy
accuracy = np.sum(y_test == predictions) / len(y_test)
print(accuracy)

# Creating odd list K for KNN
neighbors = list(range(1, len(X_train), 2))
scores = dict()

# Check the model accuracy based on the possible values for K
for K in neighbors:
    clf = KNN(k=K)
    clf.fit(X_train, y_train)
    predictions = clf.predict(X_test)
    scores[K] = np.sum(y_test == predictions) / len(y_test)

scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
