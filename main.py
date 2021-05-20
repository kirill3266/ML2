# Step 1. Import all the necessary packages
from mlxtend.plotting import plot_decision_regions
import matplotlib.pyplot as plt
import numpy as np
import sys
import tqdm
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import pandas as pd

# parse train data: read CSV files with train features (train_x) and train targets (train_y)
x_train = pd.read_csv("x_train.csv", header=None)
y_train = pd.read_csv("y_train.csv", header=None)
# show first 10 samples
pd.concat([x_train, y_train], axis=1).head(10)

# convert pandas dataframe to numpy arrays and matrices and diplay the dimensions of train dataset
x_train = x_train.to_numpy()
y_train = y_train.to_numpy()
print("Shape of train features:", x_train.shape)
print("Shape of train targets:", y_train.shape)

class LogisticRegressionGD(object):
    """Logistic Regression Classifier using gradient descent.

    Parameters
    ------------
    eta : float
      Learning rate (between 0.0 and 1.0)
    n_iter : int
      Passes over the training dataset.
    random_state : int
      Random number generator seed for random weight
      initialization.


    Attributes
    -----------
    w_ : 1d-array
      Weights after fitting.
    cost_ : list
      Logistic cost function value in each epoch.

    """

    def __init__(self, eta=0.05, n_iter=100, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state

    def fit(self, X, y):
        """ Fit training data.

        Parameters
        ----------
        X : {array-like}, shape = [n_examples, n_features]
          Training vectors, where n_examples is the number of examples and
          n_features is the number of features.
        y : array-like, shape = [n_examples]
          Target values.

        Returns
        -------
        self : object

        """
        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=1 + X.shape[1])
        self.cost_ = []

        for i in range(self.n_iter):
            net_input = self.net_input(X)
            output = self.activation(net_input)
            errors = (y - output)
            self.w_[1:] += self.eta * X.T.dot(errors)
            self.w_[0] += self.eta * errors.sum()

            # note that we compute the logistic `cost` now
            # instead of the sum of squared errors cost
            cost = -y.dot(np.log(output)) - ((1 - y).dot(np.log(1 - output)))
            self.cost_.append(cost)
        return self

    def net_input(self, X):
        """Calculate net input"""
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def activation(self, z):
        """Compute logistic sigmoid activation"""
        return 1. / (1. + np.exp(-np.clip(z, -250, 250)))

    def predict(self, X):
        """Return class label after unit step"""
        return np.where(self.net_input(X) >= 0.0, 1, 0)
        # equivalent to:
        # return np.where(self.activation(self.net_input(X)) >= 0.5, 1, 0)               >= 0.5, 1, 0)

x_train_01_subset = x_train[(y_train == 0) | (y_train == 1)]
y_train_01_subset = y_train[(y_train == 0) | (y_train == 1)]

lrgd = LogisticRegressionGD(eta=0.05, n_iter=1000, random_state=1)
lrgd.fit(x_train_01_subset,
         y_train_01_subset)

plot_decision_regions(X=x_train_01_subset,
                      y=y_train_01_subset,
                      classifier=lrgd)

plt.xlabel('petal length [standardized]')
plt.ylabel('petal width [standardized]')
plt.legend(loc='upper left')

plt.tight_layout()
#plt.savefig('images/03_05.png', dpi=300)
plt.show()


# # In this demo we will use linear regression to predict targets from features.
# # In linear regression model with parameters thetas
# # the prediction y is calculated from features x using linear combination of x and thetas.
# # For example, for the case of 2 features:
# # y = theta_0 * x_o + theta_1 * x_1
# # Let's define some helper functions
# def predict_fn(x, thetas):
#     '''
#     Predict target from features x using parameters thetas and linear regression
#     param x: vector of input features
#     param thetas: vector of linear regression parameters
#     return y_hat: predicted scalar value
#     '''
#     y_hat = np.dot(x, thetas)
#     return y_hat
#
# def loss_fn(x_train, y_train, thetas):
#     y_predicted = predict_fn(x_train, thetas)
#     loss = np.mean(np.power(y_train - y_predicted, 2))
#     return loss
#
# def gradient_fn(x_train, y_train, thetas):
#     y_predicted = predict_fn(x_train, thetas)
#     err = y_predicted - y_train
#     N = len(y_train)
#     g = np.mean(err * x_train)
#     return g
#
# # now let's find optimal parameters using gradient descent
# MAX_ITER = 100000
# thetas = np.random.randn(2, 1)
# alpha = 1e-3
# progress = tqdm.tqdm(range(MAX_ITER), "Training", file=sys.stdout)
# loss_val = loss_fn(x_train, y_train, thetas)
# progress.set_postfix(loss_val=loss_val)
# for iter in progress:
#     gradient = gradient_fn(x_train, y_train, thetas)
#     thetas = thetas - alpha*gradient
#     # TODO: add stop conditions
#     # if stop_condition is True:
#     # break
#     if iter % 100 == 0:
#         loss_val = loss_fn(x_train, y_train, thetas)
#         progress.set_postfix(loss_val=f"{loss_val:8.4f}", thetas=f"{thetas[0][0]:5.4f} {thetas[1][0]: 5.4f}")
#     progress.close()
#
# for i in range(10):
#     y_hat = predict_fn(x_train[i], thetas)
#     print("Target: ", y_train[i][0], ", predicted:", y_hat[0])
