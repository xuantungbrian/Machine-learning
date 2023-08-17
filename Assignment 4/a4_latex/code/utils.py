import argparse
from pathlib import Path
import pickle

import numpy as np
from scipy.optimize import approx_fprime

DATA_DIR = Path(__file__).parent.parent / "data"


def load_dataset(dataset_name, standardize=True, add_bias=True):
    with open((DATA_DIR / dataset_name).with_suffix(".pkl"), "rb") as f:
        data = pickle.load(f)

    X = data["X"]
    y = data["y"].astype(np.int32)
    Xvalid = data["Xvalidate"]
    yvalid = data["yvalidate"].astype(np.int32)

    if standardize:
        X, mu, sigma = standardize_cols(X)
        Xvalid, _, _ = standardize_cols(Xvalid, mu, sigma)

    if add_bias:
        X = np.hstack([np.ones((X.shape[0], 1)), X])
        Xvalid = np.hstack([np.ones((Xvalid.shape[0], 1)), Xvalid])

    return {"X": X, "y": y, "Xvalid": Xvalid, "yvalid": yvalid}

def data_split(X, y):
    n = X.shape[0]
    perm = np.random.permutation(n)
    valid_start = int(n / 2) + 1
    valid_end = n
    valid_ndx = perm[valid_start:valid_end]
    train_ndx = np.setdiff1d(np.arange(n), valid_ndx)
    Xtrain = X[train_ndx, :]
    ytrain = y[train_ndx]
    Xvalid = X[valid_ndx, :]
    yvalid = y[valid_ndx]
    return Xtrain, ytrain, Xvalid, yvalid

def standardize_cols(X, mu=None, sigma=None):
    "Standardize each column to have mean 0 and variance 1"
    n_rows, n_cols = X.shape

    if mu is None:
        mu = np.mean(X, axis=0)

    if sigma is None:
        sigma = np.std(X, axis=0)
        sigma[sigma < 1e-8] = 1.0

    return (X - mu) / sigma, mu, sigma


def check_gradient(model, X, y):
    # This checks that the gradient implementation is correct
    w = np.random.rand(*model.w.shape)
    f, g = model.fun_obj(w, X, y)

    # Check the gradient
    estimated_gradient = approx_fprime(
        w, lambda w: model.fun_obj(w, X, y)[0], epsilon=1e-6
    )

    implemented_gradient = model.fun_obj(w, X, y)[1]

    if np.max(np.abs(estimated_gradient - implemented_gradient) > 1e-4):
        raise Exception(
            "User and numerical derivatives differ:\n%s\n%s"
            % (estimated_gradient[:5], implemented_gradient[:5])
        )
    else:
        print("User and numerical derivatives agree.")


def classification_error(y, yhat):
    return np.mean(y != yhat)


def ensure_1d(x):
    if x.ndim == 1:
        return x
    elif x.ndim == 2:
        return x.squeeze(axis=1)
    elif x.ndim == 0:
        return x[np.newaxis]
    else:
        raise ValueError(f"invalid shape {x.shape} for ensure_1d")


################################################################################
# Helpers for setting up the command-line interface

_funcs = {}


def handle(number):
    def register(func):
        _funcs[number] = func
        return func

    return register


def run(question):
    if question not in _funcs:
        raise ValueError(f"unknown question {question}")
    return _funcs[question]()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("question", choices=sorted(_funcs.keys()) + ["all"])
    args = parser.parse_args()
    if args.question == "all":
        for q in sorted(_funcs.keys()):
            start = f"== {q} "
            print("\n" + start + "=" * (80 - len(start)))
            run(q)
    else:
        return run(args.question)
