#!/usr/bin/env python
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# make sure we're working in the directory this file lives in,
# for imports and for simplicity with relative paths
os.chdir(Path(__file__).parent.resolve())

from encoders import PCAEncoder
from kernels import GaussianRBFKernel, LinearKernel, PolynomialKernel
from linear_models import (
    LinearModel,
    LinearClassifier,
    KernelClassifier,
)
from optimizers import (
    GradientDescent,
    GradientDescentLineSearch,
    StochasticGradient,
)
from fun_obj import (
    LeastSquaresLoss,
    KernelLogisticRegressionLossL2,
)
from learning_rate_getters import (
    ConstantLR,
)
from utils import (
    load_dataset,
    load_trainval,
    load_and_split,
    plot_classifier,
    savefig,
    standardize_cols,
    handle,
    run,
    main,
)


@handle("1")
def q1():
    X_train, y_train, X_val, y_val = load_and_split("nonLinearData.pkl")

    # kernel logistic regression with a linear kernel
    loss_fn = KernelLogisticRegressionLossL2(1)
    optimizer = GradientDescentLineSearch()
    kernel = LinearKernel()
    klr_model = KernelClassifier(loss_fn, optimizer, kernel)
    klr_model.fit(X_train, y_train)

    print(f"Training error {np.mean(klr_model.predict(X_train) != y_train):.1%}")
    print(f"Validation error {np.mean(klr_model.predict(X_val) != y_val):.1%}")

    fig = plot_classifier(klr_model, X_train, y_train)
    savefig("logRegLinear.png", fig)


@handle("1.1")
def q1_1():
    X_train, y_train, X_val, y_val = load_and_split("nonLinearData.pkl")
    loss_fn = KernelLogisticRegressionLossL2(0.01)
    optimizer = GradientDescentLineSearch()
    kernel = PolynomialKernel(2)
    klr_model = KernelClassifier(loss_fn, optimizer, kernel)
    klr_model.fit(X_train, y_train)
    

    print(f"Training error {np.mean(klr_model.predict(X_train) != y_train):.1%}")
    print(f"Validation error {np.mean(klr_model.predict(X_val) != y_val):.1%}")
    fig = plot_classifier(klr_model, X_train, y_train)
    savefig("logRegPolynomial.png", fig)

    print()
    loss_fn = KernelLogisticRegressionLossL2(0.01)
    optimizer = GradientDescentLineSearch()
    kernel = GaussianRBFKernel(0.5)
    klr_model = KernelClassifier(loss_fn, optimizer, kernel)
    klr_model.fit(X_train, y_train)

    print(f"Training error {np.mean(klr_model.predict(X_train) != y_train):.1%}")
    print(f"Validation error {np.mean(klr_model.predict(X_val) != y_val):.1%}")
    fig = plot_classifier(klr_model, X_train, y_train)
    savefig("logRegGaussian.png", fig)

@handle("1.2")
def q1_2():
    X_train, y_train, X_val, y_val = load_and_split("nonLinearData.pkl")

    sigmas = 10.0 ** np.array([-2, -1, 0, 1, 2])
    lammys = 10.0 ** np.array([-4, -3, -2, -1, 0, 1, 2])

    # train_errs[i, j] should be the train error for sigmas[i], lammys[j]
    train_errs = np.full((len(sigmas), len(lammys)), 100.0)
    val_errs = np.full((len(sigmas), len(lammys)), 100.0)  # same for val
    i = -1
    j = 0

    for sigma in sigmas:
        i = i + 1
        j = 0
        for lammy in lammys:
            loss_fn = KernelLogisticRegressionLossL2(lammy)
            optimizer = GradientDescentLineSearch()
            kernel = GaussianRBFKernel(sigma)
            klr_model = KernelClassifier(loss_fn, optimizer, kernel)
            klr_model.fit(X_train, y_train)
            train_errs[i, j] = np.mean(klr_model.predict(X_train) != y_train)
            val_errs[i, j] = np.mean(klr_model.predict(X_val) != y_val)
            j = j + 1
    
    print(np.argmin(train_errs))
    print(np.argmin(val_errs))

    """
    
    # Make a picture with the two error arrays. No need to worry about details here.
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    norm = plt.Normalize(vmin=0, vmax=max(train_errs.max(), val_errs.max()))
    for (name, errs), ax in zip([("training", train_errs), ("val", val_errs)], axes):
        cax = ax.matshow(errs, norm=norm)

        ax.set_title(f"{name} errors")
        ax.set_ylabel(r"$\sigma$")
        ax.set_yticks(range(len(sigmas)))
        ax.set_yticklabels([str(sigma) for sigma in sigmas])
        ax.set_xlabel(r"$\lambda$")
        ax.set_xticks(range(len(lammys)))
        ax.set_xticklabels([str(lammy) for lammy in lammys])
        ax.xaxis.set_ticks_position("bottom")
    fig.colorbar(cax)
    savefig("logRegRBF_grids.png", fig)
    """


@handle("3.2")
def q3_2():
    data = load_dataset("animals.pkl")
    X_train = data["X"]
    animal_names = data["animals"]
    trait_names = data["traits"]

    # Standardize features
    X_train_standardized, mu, sigma = standardize_cols(X_train)
    n, d = X_train_standardized.shape

    # Matrix plot
    fig, ax = plt.subplots()
    ax.imshow(X_train_standardized)
    savefig("animals_matrix.png", fig)
    plt.close(fig)

    # 2D visualization
    np.random.seed(3164)  # make sure you keep this seed
    j1, j2 = np.random.choice(d, 2, replace=False)  # choose 2 random features
    random_is = np.random.choice(n, 15, replace=False)  # choose random examples

    fig, ax = plt.subplots()
    ax.scatter(X_train_standardized[:, j1], X_train_standardized[:, j2])
    for i in random_is:
        xy = X_train_standardized[i, [j1, j2]]
        ax.annotate(animal_names[i], xy=xy)
    savefig("animals_random.png", fig)
    plt.close(fig)

    #TODO YOUR CODE HERE FOR Q3.2 AND Q3.3
    model = PCAEncoder(14)
    model.fit(X_train)
    Z = model.encode(X_train)
    fig, ax = plt.subplots()
    ax.scatter(Z[:,0], Z[:,1])
    for i in range(n):
        xy = Z[i, :]
        ax.annotate(animal_names[i], xy=xy)
    savefig("PCA32.png", fig)
    
    

    #Q3.3
    """
   
    mu = np.mean(X_train_standardized, axis=0)
    X = X_train_standardized - mu
    VarEx = np.linalg.norm(Z @ model.W-X, 'fro')**2 / np.linalg.norm(X, 'fro')**2
    print(VarEx)
     """



if __name__ == "__main__":
    main()
