import numpy as np
import utils


class DecisionStumpEquality:
    """
    This is a decision stump that branches on whether the value of X is
    "almost equal to" some threshold.

    This probably isn't a thing you want to actually do, it's just an example.
    """

    y_hat_yes = None
    y_hat_no = None
    j_best = None
    t_best = None

    def fit(self, X, y):
        n, d = X.shape

        # Get an array with the number of 0's, number of 1's, etc.
        count = np.bincount(y)

        # Get the index of the largest value in count.
        # Thus, y_mode is the mode (most popular value) of y
        y_mode = np.argmax(count)

        self.y_hat_yes = y_mode
        self.y_hat_no = None
        self.j_best = None
        self.t_best = None

        # If all the labels are the same, no need to split further
        if np.unique(y).size <= 1:
            return

        minError = np.sum(y != y_mode)
        
        # Loop over features looking for the best split
        for j in range(d):
            for i in range(n):
                # Choose value to equate to
                t = np.round(X[i, j])

                # Find most likely class for each split
                is_almost_equal = np.round(X[:, j]) == t
                y_yes_mode = utils.mode(y[is_almost_equal])
                y_no_mode = utils.mode(y[~is_almost_equal])  # ~ is "logical not"

                # Make predictions
                y_pred = y_yes_mode * np.ones(n)
                y_pred[np.round(X[:, j]) != t] = y_no_mode

                # Compute error
                errors = np.sum(y_pred != y)

                # Compare to minimum error so far
                if errors < minError:
                    # This is the lowest error, store this value
                    minError = errors
                    self.j_best = j#longtitude or latitude
                    self.t_best = t#the value
                    self.y_hat_yes = y_yes_mode#the value appear the most for those that equals t
                    self.y_hat_no = y_no_mode

    def predict(self, X):
        n, d = X.shape
        X = np.round(X)

        if self.j_best is None: #if there is just 1 in y
            return self.y_hat_yes * np.ones(n)

        y_hat = np.zeros(n)

        for i in range(n):
            if X[i, self.j_best] == self.t_best:
                y_hat[i] = self.y_hat_yes
            else:
                y_hat[i] = self.y_hat_no

        return y_hat


class DecisionStumpErrorRate:
    y_hat_yes = None
    y_hat_no = None
    j_best = None
    t_best = None

    def fit(self, X, y):
        n, d = X.shape

        # Get an array with the number of 0's, number of 1's, etc.
        count = np.bincount(y)

        # Get the index of the largest value in count.
        # Thus, y_mode is the mode (most popular value) of y
        y_mode = np.argmax(count)

        self.y_hat_yes = y_mode
        self.y_hat_no = None
        self.j_best = None
        self.t_best = None

        # If all the labels are the same, no need to split further
        if np.unique(y).size <= 1:
            return

        minError = np.sum(y != y_mode)
        
        # Loop over features looking for the best split
        for j in range(d):
            for i in range(n):
                # Choose value to equate to
                t = X[i, j]

                # Find most likely class for each split
                is_almost_equal = X[:, j] > t
                y_yes_mode = utils.mode(y[is_almost_equal])
                y_no_mode = utils.mode(y[~is_almost_equal])  # ~ is "logical not"

                # Make predictions
                y_pred = y_yes_mode * np.ones(n)
                y_pred[X[:, j] <= t] = y_no_mode

                # Compute error
                errors = np.sum(y_pred != y)

                # Compare to minimum error so far
                if errors < minError:
                    # This is the lowest error, store this value
                    minError = errors
                    self.j_best = j#longtitude or latitude
                    self.t_best = t#the value
                    self.y_hat_yes = y_yes_mode#the value appear the most for those that equals t
                    self.y_hat_no = y_no_mode

    def predict(self, X):
        n, d = X.shape

        if self.j_best is None: #if there is just 1 in y
            return self.y_hat_yes * np.ones(n)

        y_hat = np.zeros(n)

        for i in range(n):
            if X[i, self.j_best] > self.t_best:
                y_hat[i] = self.y_hat_yes
            else:
                y_hat[i] = self.y_hat_no

        return y_hat


def entropy(p):
    """
    A helper function that computes the entropy of the
    discrete distribution p (stored in a 1D numpy array).
    The elements of p should add up to 1.
    This function ensures lim p-->0 of p log(p) = 0
    which is mathematically true, but numerically results in NaN
    because log(0) returns -Inf.
    """
    plogp = 0 * p  # initialize full of zeros
    plogp[p > 0] = p[p > 0] * np.log(p[p > 0])  # only do the computation when p>0
    return -np.sum(plogp)


class DecisionStumpInfoGain(DecisionStumpErrorRate):
    # This is not required, but one way to simplify the code is
    # to have this class inherit from DecisionStumpErrorRate.
    # Which methods (init, fit, predict) do you need to overwrite?
    y_hat_yes = None
    y_hat_no = None
    j_best = None
    t_best = None

    def fit(self, X, y):
        n, d = X.shape

        # Get an array with the number of 0's, number of 1's, etc.
        count = np.bincount(y)

        # Get the index of the largest value in count.
        # Thus, y_mode is the mode (most popular value) of y
        y_mode = np.argmax(count)

        self.y_hat_yes = y_mode
        self.y_hat_no = None
        self.j_best = None
        self.t_best = None

        # If all the labels are the same, no need to split further
        if np.unique(y).size <= 1:
            return

        minInfogain = 0
        
        # Loop over features looking for the best split
        for j in range(d):
            for i in range(n):
                # Choose value to equate to
                t = X[i, j]

                # Find most likely class for each split
                is_almost_equal = X[:, j] > t
                y_yes_mode = utils.mode(y[is_almost_equal])
                y_no_mode = utils.mode(y[~is_almost_equal])  # ~ is "logical not"

                y_leaf_1 = np.copy(y[is_almost_equal])
                y_leaf_2 = np.copy(y[~is_almost_equal])
                
                n_leaf_1 = y_leaf_1.size
                n_leaf_2 = y_leaf_2.size

                if (n_leaf_1 == 0):
                    child_1 = 0
                else:
                    count_leaf_1 = np.bincount(y_leaf_1, minlength = 1)/n_leaf_1
                    child_1 = n_leaf_1/n*entropy(count_leaf_1)
                
                if (n_leaf_2 == 0):
                    child_2 = 0
                else:
                    count_leaf_2 = np.bincount(y_leaf_2, minlength = 1)/n_leaf_2
                    child_2 = n_leaf_2/n*entropy(count_leaf_2)

                Infogain = entropy(count/n) - child_1 - child_2

                if Infogain > minInfogain:
                    minInfogain = Infogain
                    self.j_best = j#longtitude or latitude
                    self.t_best = t#the value
                    self.y_hat_yes = y_yes_mode#the value appear the most for those that equals t
                    self.y_hat_no = y_no_mode
