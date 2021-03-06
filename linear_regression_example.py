import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
from matplotlib import pyplot as plt
from linear_regression import OLS
plt.style.use('bmh')

# Make a sample dataset and split it into testing and training sets
X, y = datasets.make_regression(n_samples=1000, n_features=1, noise=50, random_state=1337)
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.66, random_state=1337)

# Initialize the regressor, fit the model and make a prediction
regressor = OLS(lr=0.05, n_iters=1000)
regressor.fit(X_train=X_train, y_train=y_train)
y_predicted = regressor.predict(X_test=X_test)

# Evaluate the model
mse = np.mean((y_test - y_predicted) ** 2)

# Visualize the sample data and the regression line
fig, ax = plt.subplots()
ax.scatter(X, y, label='Data Points')
ax.plot(X_test, y_predicted, color='red', label='Fitted Line')
ax.set_facecolor('white')
ax.set_xlabel('Features Values', size=15)
ax.set_ylabel('Target Values', size=15)
fig.legend(loc='upper right', bbox_to_anchor=(0.85, 0.99))
ax.tick_params(axis='both', labelsize=12)
plt.tight_layout()
plt.savefig(r"/Users/dimitar/IdeaProjects/ml_from_scratch/ols.png")
plt.show()

