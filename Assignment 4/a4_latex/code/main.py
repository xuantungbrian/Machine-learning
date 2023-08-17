#!/usr/bin/env python
import os
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

# make sure we're working in the directory this file lives in,
# for imports and for simplicity with relative paths
os.chdir(Path(__file__).parent.resolve())

from fun_obj import (
    LogisticRegressionLoss,
    LogisticRegressionLossL0,
    LogisticRegressionLossL2,
    SoftmaxLoss,
)
import linear_models
from optimizers import (
    GradientDescentLineSearch,
    GradientDescentLineSearchProxL1,
)
from utils import load_dataset, data_split, classification_error, handle, run, main

@handle("1")
def q1():
    data = load_dataset("basisData",standardize=False, add_bias=False)
    X, y = data["X"], data["y"]
    X_test, y_test = data["Xvalid"], data["yvalid"]
    X_train, y_train, X_valid, y_valid = data_split(X,y)

    minErr = np.inf
    bestSigma = None
    for sigma in [2.0**i for i in range(-15, 16)]:

        # Train on the training set
        model = linear_models.LeastSquaresRBF(sigma)
        model.fit(X_train, y_train)
        # Compute the error on the validation set
        yhat = model.predict(X_valid)
        validError = np.mean((yhat - y_valid)**2)
        print("With sigma = {:.3f}, validError = {:.2f}".format(sigma, validError))

        # Keep track of the lowest validation error
        if validError < minErr:
            minErr = validError
            bestSigma = sigma

    # Now fit the model based on the full dataset
    model= linear_models.LeastSquaresRBF(bestSigma)
    model.fit(X, y)
    # Report the error on the test set
    yhat = model.predict(X_test)
    testError = np.mean((yhat - y_test)**2)
    print("With best sigma of {:.3f}, testError = {:.2f}".format(bestSigma, testError))

    # Plot model
    X = X.flatten()
    y = y.flatten()
    X_test = X_test.flatten()
    y_test = y_test.flatten()
    plt.scatter(X, y, marker='.', label='Training Data')
    plt.scatter(X_test, y_test, marker='.', label='Test Data')
    Xhat = np.arange(np.min(X), np.max(X) + 0.1, 0.1).reshape(-1, 1)
    yhat = model.predict(Xhat)
    plt.plot(Xhat, yhat)
    plt.show()

@handle("1.1")
def q1_1():
    data = load_dataset("basisData",standardize=False, add_bias=False)
    X, y = data["X"], data["y"]
    X_test, y_test = data["Xvalid"], data["yvalid"]

    """YOUR CODE HERE FOR Q1.1"""
    raise NotImplementedError

@handle("1.2")
def q1_2():
    data = load_dataset("basisData",standardize=False, add_bias=False)
    X, y = data["X"], data["y"]
    X_test, y_test = data["Xvalid"], data["yvalid"]

    """YOUR CODE HERE FOR Q1.2"""
    raise NotImplementedError

@handle("2")
def q2():
    data = load_dataset("logisticData")
    X, y = data["X"], data["y"]
    X_valid, y_valid = data["Xvalid"], data["yvalid"]

    fun_obj = LogisticRegressionLoss()
    optimizer = GradientDescentLineSearch(max_evals=400, verbose=False)
    model = linear_models.LinearClassifier(fun_obj, optimizer)
    model.fit(X, y)

    train_err = classification_error(model.predict(X), y)
    print(f"Linear Training error: {train_err:.3f}")

    val_err = classification_error(model.predict(X_valid), y_valid)
    print(f"Linear Validation error: {val_err:.3f}")

    print(f"# nonZeros: {np.sum(model.w != 0)}")
    print(f"# function evals: {optimizer.num_evals}")


@handle("2.1")
def q2_1():
    data = load_dataset("logisticData")
    X, y = data["X"], data["y"]
    X_valid, y_valid = data["Xvalid"], data["yvalid"]

    fun_obj = LogisticRegressionLossL2(1)
    optimizer = GradientDescentLineSearch(max_evals=400, verbose=False)
    model = linear_models.LinearClassifier(fun_obj, optimizer)
    model.fit(X, y)

    train_err = classification_error(model.predict(X), y)
    print(f"Linear Training error: {train_err:.3f}")

    val_err = classification_error(model.predict(X_valid), y_valid)
    print(f"Linear Validation error: {val_err:.3f}")

    print(f"# nonZeros: {np.sum(model.w != 0)}")
    print(f"# function evals: {optimizer.num_evals}")


@handle("2.2")
def q2_2():
    data = load_dataset("logisticData")
    X, y = data["X"], data["y"]
    X_valid, y_valid = data["Xvalid"], data["yvalid"]

    """YOUR CODE HERE FOR Q2.2"""
    raise NotImplementedError


@handle("2.3")
def q2_3():
    data = load_dataset("logisticData")
    X, y = data["X"], data["y"]
    X_valid, y_valid = data["Xvalid"], data["yvalid"]

    local_loss = LogisticRegressionLoss()
    global_loss = LogisticRegressionLossL0(1)
    optimizer = GradientDescentLineSearch(max_evals=400, verbose=False)
    model = linear_models.LinearClassifierForwardSel(local_loss, global_loss, optimizer)
    model.fit(X, y)

    train_err = classification_error(model.predict(X), y)
    print(f"Linear training 0-1 error: {train_err:.3f}")

    val_err = classification_error(model.predict(X_valid), y_valid)
    print(f"Linear validation 0-1 error: {val_err:.3f}")

    print(f"# nonZeros: {np.sum(model.w != 0)}")
    print(f"total function evaluations: {model.total_evals:,}")


@handle("3")
def q3():
    data = load_dataset("multiData")
    X, y = data["X"], data["y"]
    X_valid, y_valid = data["Xvalid"], data["yvalid"]

    model = linear_models.LeastSquaresClassifier()
    model.fit(X, y)

    train_err = classification_error(model.predict(X), y)
    print(f"LeastSquaresClassifier training 0-1 error: {train_err:.3f}")

    val_err = classification_error(model.predict(X_valid), y_valid)
    print(f"LeastSquaresClassifier validation 0-1 error: {val_err:.3f}")

    print(f"model predicted classes: {np.unique(model.predict(X))}")

@handle("3.3")
def q3_3():
    data = load_dataset("multiData")
    X, y = data["X"], data["y"]
    X_valid, y_valid = data["Xvalid"], data["yvalid"]

    fun_obj = SoftmaxLoss()
    optimizer = GradientDescentLineSearch(max_evals=1_000, verbose=True)
    model = linear_models.MulticlassLinearClassifier(fun_obj, optimizer)
    model.fit(X, y)

    train_err = classification_error(model.predict(X), y)
    print(f"SoftmaxLoss training 0-1 error: {train_err:.3f}")

    val_err = classification_error(model.predict(X_valid), y_valid)
    print(f"SoftmaxLoss validation 0-1 error: {val_err:.3f}")

    print(f"model predicted classes: {np.unique(model.predict(X))}")

if __name__ == "__main__":
    main()
