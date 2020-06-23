from sklearn import datasets
from sklearn.model_selection import train_test_split
import numpy as np
from logistic_regression import Logit

# Load the data and split it into training and test sets
bc = datasets.load_breast_cancer()
X, y = bc.data, bc.target
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.66, random_state=1337)

# Initialize the regressor, fit the model, and make a prediction
regressor = Logit(lr=0.01, n_iters=1000)
regressor.fit(X_train=X_train, y_train=y_train)
y_pred = regressor.predict(X_test=X_test)

# Evaluate the accuracy of the model
accuracy = np.sum(y_test == y_pred) / len(y_test)
