import numpy as np


class Logit:

    def __init__(self, lr=0.01, n_iters=1000):
        self.w = None
        self.b = None
        self.lr = lr
        self.n_iters = n_iters

    def fit(self, X_train, y_train):
        # Initialize parameters
        n_samples, n_features = X_train.shape
        self.w = np.zeros(n_features)
        self.b = 0

        # Gradient descent
        for iteration in range(self.n_iters):
            linear_model = np.dot(X_train, self.w) + self.b
            y_predicted = self._sigmoid(linear_model)

            dw = (1/n_samples) * np.dot(-2 * X_train.T, (y_train - y_predicted))
            db = (2/n_samples) * np.sum(y_train - y_predicted)

            self.w = self.w - self.lr * dw
            self.b = self.b - self.lr * db

    def predict(self, X_test):
        # Make a linear prediction
        linear_model = np.dot(X_test, self.w) + self.b
        # Transform the linear prediction via the sigmoid function
        y_predicted = self._sigmoid(linear_model)
        # Turn the predicted probabilities into classes
        y_predicted_cls = [1 if obs > 0.5 else 0 for obs in y_predicted]

        return np.array(y_predicted_cls)

    @staticmethod
    def _sigmoid(x):
        return np.float128(1/(1 + np.exp(-x)))
