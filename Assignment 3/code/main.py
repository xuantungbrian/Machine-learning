#!/usr/bin/env python
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


# make sure we're working in the directory this file lives in,
# for imports and for simplicity with relative paths
os.chdir(Path(__file__).parent.resolve())

# our code 
from optimizers import GradientDescent
from vector_quantization import VectorQuantization
from fun_obj import LeastSquaresLoss, RobustRegressionLoss
from linear_models import LeastSquares, LeastSquaresBias, LeastSquaresPoly, WeightedLeastSquares, LinearModel
from utils import load_dataset, test_and_plot, handle, run, main

@handle("1")
def q1():
    nBits = [1,2,4,6]
    for i in nBits:
        y, W, nRows, nCols = VectorQuantization.quantizeImage('dog.png',i)
        I = VectorQuantization.deQuantizeImage(y, W, nRows, nCols)
        Image.fromarray(np.uint8(I)).save("../figs/dog_" + str(i) + ".png")
        
@handle("3")
def q3():
    data = load_dataset("outliersData.pkl")
    X = data["X"]
    y = data["y"].squeeze(1)

    # Fit least-squares estimator
    model = LeastSquares()
    model.fit(X, y)
    print(model.w)

    test_and_plot(
        model, X, y, title="Least Squares", filename="least_squares_outliers.pdf"
    )

@handle("3.1")
def q3_1():
    data = load_dataset("outliersData.pkl")
    X = data["X"]
    y = data["y"].squeeze(1)

    model = WeightedLeastSquares()
    v = np.ones(500)
    v[400:] = 0.1
    print(v[399])
    model.fit(X, y, v)
    print(model.w)

    test_and_plot(
        model, X, y, title="Weighted Least Squares", filename="weighted_least_squares_outliers.pdf"
    )

@handle("3.4")
def q3_4():
    # loads the data in the form of dictionary
    data = load_dataset("outliersData.pkl")
    X = data["X"]
    y = data["y"].squeeze(1)

    fun_obj = LeastSquaresLoss()
    optimizer = GradientDescent(max_evals=100, verbose=False)
    model = LinearModel(fun_obj, optimizer)
    model.fit(X, y)
    print(model.w)

    test_and_plot(
        model,
        X,
        y,
        title="Linear Regression with Gradient Descent",
        filename="least_squares_gd.pdf",
    )


@handle("3.4.1")
def q3_4_1():
    data = load_dataset("outliersData.pkl")
    X = data["X"]
    y = data["y"].squeeze(1)

    fun_obj = RobustRegressionLoss(epsilon=1)
    optimizer = GradientDescent(max_evals=100, verbose=False)
    model = LinearModel(fun_obj, optimizer)
    model.fit(X, y)
    print(model.w)

    test_and_plot(
        model,
        X,
        y,
        title="Robust Regression with Gradient Descent",
        filename="robust_regression_gd.pdf",
    )


@handle("4")
def q4():
    data = load_dataset("basisData.pkl")
    X = data["X"]
    y = data["y"].squeeze(1)
    X_valid = data["Xtest"]
    y_valid = data["ytest"].squeeze(1)

    # Fit least-squares model
    model = LeastSquares()
    model.fit(X, y)

    test_and_plot(
        model,
        X,
        y,
        X_valid,
        y_valid,
        title="Least Squares, no bias",
        filename="least_squares_no_bias.pdf",
    )


@handle("4.1")
def q4_1():
    data = load_dataset("basisData.pkl")
    X = data["X"]
    y = data["y"].squeeze(1)
    X_valid = data["Xtest"]
    y_valid = data["ytest"].squeeze(1)

    model = LeastSquaresBias()
    model.fit(X, y)

    test_and_plot(
        model,
        X,
        y,
        X_valid,
        y_valid,
        title="Least Squares, yes bias",
        filename="least_squares_yes_bias.pdf",
    )


@handle("4.2")
def q4_2():
    data = load_dataset("basisData.pkl")
    X = data["X"]
    y = data["y"].squeeze(1)
    X_valid = data["Xtest"]
    y_valid = data["ytest"].squeeze(1)

    p_vals = [0, 1, 2, 3, 4, 5, 10, 20, 30, 50, 75, 100]
    num_runs = len(p_vals)
    err_trains = np.zeros(num_runs)
    err_valids = np.zeros(num_runs)

    plot_grid_size1 = int(np.ceil(np.sqrt(num_runs)))
    plot_grid_size2 = int(np.ceil(num_runs / plot_grid_size1))

    fig, axes = plt.subplots(
        plot_grid_size1,
        plot_grid_size2,
        figsize=(30, 20),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    for i, (p, ax) in enumerate(zip(p_vals, (ax for row in axes for ax in row))):
        print(f"p = {p}")

        model = LeastSquaresPoly(p)
        model.fit(X, y)
        y_hat = model.predict(X)
        err_train = np.mean((y_hat - y) ** 2)
        err_trains[i] = err_train

        y_hat = model.predict(X_valid)
        err_valid = np.mean((y_hat - y_valid) ** 2)
        err_valids[i] = err_valid

        ax.scatter(X, y, color="b", s=2)
        Xgrid = np.linspace(np.min(X_valid), np.max(X_valid), 1000)[:, None]
        ygrid = model.predict(Xgrid)
        ax.plot(Xgrid, ygrid, color="r")
        ax.set_title(f"p={p}")
        ax.set_ylim(np.min(y), np.max(y))

    filename = Path("..", "figs", "polynomial_fits.pdf")
    print("Saving to", filename)
    fig.savefig(filename)

    # Plot error curves
    plt.figure()
    plt.plot(p_vals, err_trains, marker="o", label="training error")
    plt.plot(p_vals, err_valids, marker="o", label="validation error")
    plt.xlabel("Degree of polynomial")
    plt.ylabel("Error")
    plt.yscale("log")
    plt.legend()
    filename = Path("..", "figs", "polynomial_error_curves.pdf")
    print("Saving to", filename)
    plt.savefig(filename)


if __name__ == "__main__":
    main()
