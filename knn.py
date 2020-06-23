import numpy as np
from collections import Counter


class KNN:

    @staticmethod
    def _get_distance(a, b):
        """Computes the Euclidian distance between two points on a plane"""
        return np.sqrt(np.sum((a - b) ** 2))

    def __init__(self, k=None):
        self.k = k

    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    def predict(self, X_test):
        """
        Makes predictions for each sample in the test set
        :param X_test: array like
        An array with the features of the testing set
        :return: numpy array
        An array with the predicted labels for each feature in the testing set
        """
        predicted_label = [self._predict(x_test) for x_test in X_test]

        return np.array(predicted_label)

    def _predict(self, x):
        """
        Makes a prediction based on the most common class among the K most common classes
        :param x: array like; shaped (n,)
        An array with one line of the values of the features
        :return: most_common_class: int
        The integer label of the most common class
        """
        # Compute the distance between x and each data point in X_train
        distances = [self._get_distance(x, x_train) for x_train in self.X_train]
        # Get the labels of the k nearest samples to x based on the distances
        k_nearest_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = [self.y_train[idx] for idx in k_nearest_indices]
        # Determine the most common of the k nearest labels
        most_common_class = Counter(k_nearest_labels).most_common(1)[0][0]

        return most_common_class


