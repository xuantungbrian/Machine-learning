import numpy as np
from numpy.linalg import norm

"""
Implementation of optimizers, following the design pattern of PyTorch,
a popular library for differentiable programming and optimization.

Optimizers are used with function objects. See fun_obj.py.
"""


class Optimizer:
    def step(self):
        pass  # this is because it's a base class, it will be implemented below.

    def set_learning_rate(self, learning_rate):
        self.learning_rate = learning_rate

    def set_parameters(self, parameters):
        self.parameters = parameters

    def set_fun_obj(self, fun_obj):
        self.fun_obj = fun_obj

    def set_fun_obj_args(self, *fun_obj_args):
        self.fun_obj_args = fun_obj_args

    def reset(self):
        """
        In case we want to re-run the optimization with different parameters, etc.
        """
        pass  # this is because it's a base class, it will be implemented below.

    def clear(self):
        """
        Soft reset, which preserves other properties but clears cached information for reuse
        """
        pass  # this is because it's a base class, it will be implemented below.


class GradientDescent(Optimizer):
    """
    Vanilla gradient descent algorithm, implemented into an Optimizer class
    """

    def __init__(
        self, optimal_tolerance=1e-2, learning_rate=1e-3, max_evals=100, verbose=False
    ):
        """
        Optimizer and function object are theoretically orthogonal,
        so the fit() methods should ideally associate the two,
        rather than have a redundant association within the constructor.
        """
        self.parameters = None
        self.optimal_tolerance = optimal_tolerance
        self.learning_rate = learning_rate
        self.initial_learning_rate = learning_rate  # for resetting
        self.max_evals = max_evals
        self.num_evals = 0
        self.verbose = verbose

        # Keep f and g as state variables to reduce redundancy
        self.f_old = None
        self.g_old = None

    def reset(self):
        """
        The state of the optimizer is tied to the state of the parameters.
        Resetting an optimizer will revert its state to the original.
        In practice, it doesn't matter whether you use reset() or initialize a new optimizer,
        but for A4 we will re-use the same optimizer, because it's more convenient and generic.
        """
        self.num_evals = 0
        self.parameters = None
        self.fun_obj_args = None
        self.learning_rate = self.initial_learning_rate
        self.f_old = None
        self.g_old = None

    def step(self):
        """
        step() does not have any argument because the parameters for optimization
        are registered via the constructor with the "parameters" argument.
        Calling optimizer.step() will conduct one step of gradient descent, i.e.
        w^{t+1} = w^t - \alpha^t * \nabla f(w^t)
        """

        if self.fun_obj is None:
            raise RuntimeError(
                "You must set the function object for the optimizer with set_fun_obj() before calling step()."
            )

        if self.parameters is None:
            raise RuntimeError(
                "You must set the parameters for the optimizer with set_parameters() before calling step()."
            )

        if self.fun_obj_args is None:
            raise RuntimeError(
                "You must set the arguments for the function object with set_fun_obj_args() before calling step()."
            )

        # Evaluate old value and gradient
        if self.f_old is None or self.g_old is None:
            self.f_old, self.g_old = self.get_function_value_and_gradient(
                self.parameters
            )

        # Perform a step: learning rate tuning and gradient descent in one call
        # This is to reduce the number of evaluations by re-using solutions from line search
        w_new, f_new, g_new = self.get_learning_rate_and_step(self.f_old, self.g_old)
        self.parameters = w_new

        # Update optimizer state for faster compute
        self.f_old = f_new
        self.g_old = g_new

        self.num_evals += 1
        break_yes = self.break_yes(g_new)
        return f_new, g_new, self.parameters, break_yes

    def get_learning_rate_and_step(self, f_old, g_old):
        """
        For vanilla gradient descent, combining learning rate and step doesn't
        necessarily give us speedup, but for backtracking line search, we can cut down
        at least one gradient computation by returning the last-used f and g values during backtracking
        """
        w_old = self.parameters
        alpha = self.learning_rate
        w_new = w_old - alpha * g_old
        f_new, g_new = self.get_function_value_and_gradient(w_new)
        return w_new, f_new, g_new

    def break_yes(self, g):
        gradient_norm = norm(g, float("inf"))
        if gradient_norm < self.optimal_tolerance:
            if self.verbose:
                print(
                    "Problem solved up to optimality tolerance {:.3f}".format(
                        self.optimal_tolerance
                    )
                )
            return True
        elif self.num_evals >= self.max_evals:
            if self.verbose:
                print(
                    "Reached maximum number of function evaluations {:.3f}".format(
                        self.max_evals
                    )
                )
            return True
        else:
            return False

    def get_next_parameter_value(self, alpha, g):
        """
        Get the new parameter value after the gradient descent step.
        Does not mutate self.parameters. step() will call this and then
        overwrite the values explicitly.
        """
        return self.parameters - alpha * g

    def get_function_value_and_gradient(self, w):
        """
        Evaluate function and gradient based on the input w.
        w is not necessarily the current parameter value.
        For vanilla gradient descent and line search, this is simply pass-through.
        For proximal and more advanced gradient methods, additional terms are introduced.
        """
        return self.fun_obj.evaluate(w, *self.fun_obj_args)

    def clear(self):
        """
        For correct implementation of stochastic gradient descent,
        clear the cached f and g values.
        """
        self.f_old = None
        self.g_old = None