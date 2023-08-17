import numpy as np
from numpy.linalg import solve
from scipy.spatial.distance import cdist

class LinearModel:
    """
    Generic linear model, supporting generic loss functions (FunObj subclasses)
    and optimizers.

    See optimizers.py for optimizers.
    See fun_obj.py for loss function objects, which must implement evaluate()
    and return f and g values corresponding to current parameters.
    """

    def __init__(self, loss_fn, optimizer, check_correctness=False):
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.bias_yes = True
        self.check_correctness = check_correctness

        # For debugging and making learning curves
        self.fs = []
        self.nonzeros = []
        self.ws = []

    def optimize(self, w_init, X, y):
        "Perform gradient descent using the optimizer."
        n, d = X.shape

        # Initial guess
        w = np.copy(w_init)
        f, g = self.loss_fn.evaluate(w, X, y)

        # Reset the optimizer state and tie it to the new parameters.
        # See optimizers.py for why reset() is useful here.
        self.optimizer.reset()
        self.optimizer.set_fun_obj(self.loss_fn)
        self.optimizer.set_parameters(w)
        self.optimizer.set_fun_obj_args(X, y)

        # Collect training information for debugging
        fs = [f]
        gs = [g]
        ws = []

        # Use gradient descent to optimize w
        while True:
            f, g, w, break_yes = self.optimizer.step()
            fs.append(f)
            gs.append(g)
            ws.append(w)
            if break_yes:
                break

        return w, fs, gs, ws

    def fit(self, X, y):
        """
        Generic fitting subroutine:
        1. Make initial guess
        2. Check correctness of function object
        3. Use gradient descent to optimize
        """
        n, d = X.shape

        # Correctness check
        if self.check_correctness:
            w = np.random.rand(d)
            self.loss_fn.check_correctness(w, X, y)

        # Initial guess
        w = np.zeros(d)

        # Optimize
        self.w, self.fs, self.gs, self.ws = self.optimize(w, X, y)

    def predict(self, X):
        """
        By default, implement linear regression prediction
        """
        return X @ self.w

class LeastSquaresRBF:
    "Least Squares with RBF basis"

    def __init__(self, sigma):
        self.sigma = sigma

    def fit(self, X, y):
        n, d = X.shape
        self.X_fit_ = X
        Z = self._RBF_basis(X, X, self.sigma)
        self.w = solve(Z.T @ Z, Z.T @ y)

    def predict(self, X_pred):
        n, d = X_pred.shape
        Z = self._RBF_basis(X_pred,self.X_fit_ ,self.sigma)
        return Z @ self.w

    # Private helper function that computes the RBF kernel matrix
    # between two sets of input data points, X1 and X2, given the kernel width sigma.
    # Returns the matrix K that is RBF kernel matrix.
    def _RBF_basis(self, X1, X2, sigma):
        K = cdist(X1, X2, 'euclidean')
        K = np.exp(-K**2 / (2 * sigma**2))
        return K

class RegularizedRBF:
    "L2-Regularized RBF Regression"

    def __init__(self, sigma, lambd):
        self.sigma = sigma
        self.lambd = lambd

    def fit(self, X, y):
        """YOUR CODE HERE FOR Q1.1"""

    def predict(self, X_pred):
        """YOUR CODE HERE FOR Q1.1"""

    # Private helper function that computes the RBF kernel matrix
    # between two sets of input data points, X1 and X2, given the kernel width sigma.
    # Returns the matrix K that is RBF kernel matrix.
    def _RBF_basis(self, X1, X2, sigma):
        K = cdist(X1, X2, 'euclidean')
        K = np.exp(-K**2 / (2 * sigma**2))
        return K
        


class LinearClassifier(LinearModel):
    """
    Generic Logistic Regression classifier,
    whose behaviour is selected by the combination of
    function object and optimizer.
    """

    def predict(self, X):
        return np.sign(X @ self.w)


class LinearClassifierForwardSel(LinearClassifier):
    """
    A logistic regression classifier that trains with forward selection.
    A gradient-based optimizer as well as an objective function is needed.
    """

    def __init__(
        self, local_loss_fn, global_loss_fn, optimizer, check_correctness=False
    ):
        """
        NOTE: There are two loss function objects involved:
        1. local_loss_fn: the loss that we optimize "inside the loop"
        1. global_loss_fn: the forward selection criterion to compare feature sets
        """
        super().__init__(
            loss_fn=local_loss_fn,
            optimizer=optimizer,
            check_correctness=check_correctness,
        )
        self.global_loss_fn = global_loss_fn

    def fit(self, X, y):
        n, d = X.shape

        # Maintain the set of selected indices, as a boolean mask array.
        # We assume that feature 0 is a bias feature, and include it by default.
        selected = np.zeros(d, dtype=bool)
        selected[0] = True
        min_loss = np.inf
        self.total_evals = 0

        # We will hill-climb until a local discrete minimum is found.
        while not np.all(selected):
            old_loss = min_loss
            print(f"Epoch {selected.sum():>3}:", end=" ")

            for j in range(d):
                if selected[j]:
                    continue

                selected_with_j = selected.copy()
                selected_with_j[j] = True

                """YOUR CODE HERE FOR Q2.3"""
                # TODO: Fit the model with 'j' added to the features,
                # then compute the loss and update the min_loss/best_feature.
                # Also update self.total_evals.
                raise NotImplementedError

            if min_loss < old_loss:  # something in the loop helped our model
                selected[best_feature] = True
                print(f"adding feature {best_feature:>3} - loss {min_loss:>7.3f}")
            else:
                print("nothing helped to add; done.")
                break
        else:  # triggers if we didn't break out of the loop
            print("wow, we selected everything")

        w_init = np.zeros(selected.sum())
        w_on_sub, *_ = self.optimize(w_init, X[:, selected], y)
        self.total_evals += self.optimizer.num_evals

        self.w = np.zeros(d)
        self.w[selected] = w_on_sub


class LeastSquaresClassifier:
    """
    Uses the normal equations to fit a one-vs-all least squares classifier.
    """

    def fit(self, X, y):
        n, d = X.shape
        self.n_classes = np.unique(y).size

        # Initial guess
        self.W = np.zeros((self.n_classes, d))

        for i in range(self.n_classes):
            ytmp = y.copy().astype(float)
            ytmp[y == i] = 1
            ytmp[y != i] = -1

            # solve the normal equations
            # with a bit of regularization for numerical reasons
            self.W[i] = np.linalg.solve(X.T @ X + 0.0001 * np.eye(d), X.T @ ytmp)

    def predict(self, X_hat):
        return np.argmax(X_hat @ self.W.T, axis=1)


class MulticlassLinearClassifier(LinearClassifier):
    """
    LinearClassifier's extention for multiclass classification.
    The constructor method and optimize() are inherited, so
    all you need to implement are fit() and predict() methods.
    """

    def fit(self, X, y):
        """YOUR CODE HERE FOR Q3.3"""
        raise NotImplementedError
        self.W = W

    def predict(self, X_hat):
        """YOUR CODE HERE FOR Q3.3"""
        raise NotImplementedError
