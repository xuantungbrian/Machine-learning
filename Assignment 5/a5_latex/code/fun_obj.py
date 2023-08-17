import numpy as np
from scipy.optimize.optimize import approx_fprime
from scipy.special import logsumexp

from utils import ensure_1d

"""
Implementation of function objects.
Function objects encapsulate the behaviour of an objective function that we optimize.
Simply put, implement evaluate(w, X, y) to get the numerical values corresponding to:
f, the function value (scalar) and
g, the gradient (vector).

Function objects are used with optimizers to navigate the parameter space and
to find the optimal parameters (vector). See optimizers.py.
"""


class FunObj:
    """
    Function object for encapsulating evaluations of functions and gradients
    """

    def evaluate(self, w, X, y):
        """
        Evaluates the function AND its gradient w.r.t. w.
        Returns the numerical values based on the input.
        IMPORTANT: w is assumed to be a 1d-array, hence shaping will have to be handled.
        """
        raise NotImplementedError("This is a base class, don't call this")

    def check_correctness(self, w, X, y):
        n, d = X.shape
        estimated_gradient = approx_fprime(
            w, lambda w: self.evaluate(w, X, y)[0], epsilon=1e-6
        )
        _, implemented_gradient = self.evaluate(w, X, y)
        difference = estimated_gradient - implemented_gradient
        if np.max(np.abs(difference) > 1e-4):
            print(
                "User and numerical derivatives differ: %s vs. %s"
                % (estimated_gradient, implemented_gradient)
            )
        else:
            print("User and numerical derivatives agree.")


class LeastSquaresLoss(FunObj):
    def evaluate(self, w, X, y):
        """
        Evaluates the function and gradient of least squares objective.
        Least squares objective is half the sum of squared residuals.
        """
        # help avoid mistakes by potentially reshaping our arguments
        w = ensure_1d(w)
        y = ensure_1d(y)

        y_hat = X @ w
        m_residuals = y_hat - y  # minus residuals, slightly more convenient here

        # Loss is sum of squared residuals
        f = 0.5 * np.sum(m_residuals ** 2)

        # The gradient, derived mathematically then implemented here
        g = X.T @ m_residuals  # X^T X w - X^T y

        return f, g


class LeastSquaresLossL2(LeastSquaresLoss):
    def __init__(self, lammy):
        self.lammy = lammy

    def evaluate(self, w, X, y):
        f_base, g_base = super().evaluate(w, X, y)
        f = f_base + self.lammy / 2 * (w @ w)
        g = g_base + self.lammy * w
        return f, g


class RobustRegressionLoss(FunObj):
    def evaluate(self, w, X, y):
        """
        Evaluates the function and gradient of ROBUST least squares objective.
        """
        # help avoid mistakes by potentially reshaping our arguments
        w = ensure_1d(w)
        y = ensure_1d(y)

        y_hat = X @ w
        residuals = y - y_hat
        exp_residuals = np.exp(residuals)
        exp_minuses = np.exp(-residuals)

        f = np.sum(np.log(exp_minuses + exp_residuals))

        # s is the negative of the "soft sign"
        s = (exp_minuses - exp_residuals) / (exp_minuses + exp_residuals)
        g = X.T @ s

        return f, g


class LogisticRegressionLoss(FunObj):
    def evaluate(self, w, X, y):
        """
        Evaluates the function and gradient of logistics regression objective.
        """
        # help avoid mistakes by potentially reshaping our arguments
        w = ensure_1d(w)
        y = ensure_1d(y)

        Xw = X @ w
        yXw = y * Xw  # element-wise multiply; the y_i are in {-1, 1}

        # Calculate the function value
        # logaddexp(a, b) = log(exp(a) + exp(b)), but more numerically stable
        f = np.logaddexp(0, -yXw).sum()

        # Calculate the gradient value
        with np.errstate(over="ignore"):  # overflowing here is okay: we get 0
            g_bits = -y / (1 + np.exp(yXw))
        g = X.T @ g_bits

        # 1 / (1 + exp(yXw)) = exp(-yXw) / (1 + exp(-yXw))

        # X.T @ (-y / (1 + exp(y * X w)))
        #

        return f, g


class KernelLogisticRegressionLossL2(FunObj):
    def __init__(self, lammy):
        self.lammy = lammy

    def evaluate(self, u, K, y):
        """
        Here u is the length-n vector defining our linear combination in
        the (potentially infinite-dimensional) Z space,
        and K is the Gram matrix K[i, i'] = k(x_i, x_{i'}).

        Note the L2 regularizer is in the transformed space too, not on u.
        """
        u = ensure_1d(u)
        y = ensure_1d(y)

        yKu = y * (K @ u)

        f = np.logaddexp(0, -yKu).sum() + (self.lammy / 2) * u @ K @ u

        with np.errstate(over="ignore"):  # overflowing here is okay: we get 0
            g_bits = -y / (1 + np.exp(yKu))
        g = K @ g_bits + self.lammy * K @ u

        return f, g


class LogisticRegressionLossL0(FunObj):
    def __init__(self, lammy):
        self.lammy = lammy

    def evaluate(self, w, X, y):
        """
        Evaluates the function value of of L0-regularized logistics regression
        objective.
        """
        w = ensure_1d(w)
        y = ensure_1d(y)

        Xw = X @ w
        yXw = y * Xw  # element-wise multiply; the y should be in {-1, 1}

        # Calculate the function value
        f = np.logaddexp(0, -yXw).sum() + self.lammy * np.sum(w != 0)

        # We cannot differentiate the "length" function
        g = None
        return f, g
