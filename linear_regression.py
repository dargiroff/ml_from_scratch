import numpy as np


class OLS:

    def __init__(self, lr=0.001, n_iters=1000):
        self.w = None
        self.b = None
        self.lr = lr
        self.n_iters = n_iters

    def fit(self, X_train, y_train):
        """Fit the model via a gradient descent"""
        # Initialize the parameters
        n_samples, n_features = X_train.shape
        self.w = np.zeros(n_features)
        self.b = 0

        for iteration in range(self.n_iters):
            y_predicted = np.dot(X_train, self.w) + self.b

            # Calculate the derivatives
            dw = (1/n_samples) * np.dot(-2 * X_train.T, (y_train - y_predicted))
            db = (-2/n_samples) * np.sum(y_train - y_predicted)

            # Update the weights and the bias according to the learning rate and the derivatives
            self.w = self.w - self.lr * dw
            self.b = self.b - self.lr * db

    def predict(self, X_test):
        return np.dot(X_test, self.w) + self.b
