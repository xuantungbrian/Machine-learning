#!/usr/bin/env python
import argparse
import os
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

# make sure we're working in the directory this file lives in,
# for imports and for simplicity with relative paths
os.chdir(Path(__file__).parent.resolve())

# our code
from utils import load_dataset, plot_classifier, handle, run, main
from decision_stump import DecisionStumpInfoGain
from decision_tree import DecisionTree
from kmeans import Kmeans
from knn import KNN
from naive_bayes import NaiveBayes, NaiveBayesLaplace
from random_tree import RandomForest, RandomTree



@handle("1.2")
def q1_2():
    dataset = load_dataset("newsgroups.pkl")

    X = dataset["X"].astype(bool)
    y = dataset["y"]
    X_valid = dataset["Xvalidate"]
    y_valid = dataset["yvalidate"]
    groupnames = dataset["groupnames"]
    wordlist = dataset["wordlist"]
    names = X[72]
    """YOUR CODE HERE FOR Q1.2"""
    print(wordlist[72])
    print(wordlist[X[802]==True])
    print(groupnames[y[802]])


@handle("1.3")
def q1_3():
    dataset = load_dataset("newsgroups.pkl")

    X = dataset["X"]
    y = dataset["y"]
    X_valid = dataset["Xvalidate"]
    y_valid = dataset["yvalidate"]

    print(f"d = {X.shape[1]}")
    print(f"n = {X.shape[0]}")
    print(f"t = {X_valid.shape[0]}")
    print(f"Num classes = {len(np.unique(y))}")

    model = NaiveBayes(num_classes=4)
    model.fit(X, y)

    y_hat = model.predict(X)
    err_train = np.mean(y_hat != y)
    print(f"Naive Bayes training error: {err_train:.3f}")

    y_hat = model.predict(X_valid)
    err_valid = np.mean(y_hat != y_valid)
    print(f"Naive Bayes validation error: {err_valid:.3f}")

@handle("2.1")
def q2_1():
    dataset = load_dataset("citiesSmall.pkl")

    X = dataset["X"]
    y = dataset["y"]
    X_test = dataset["Xtest"]
    y_test = dataset["ytest"]

    """YOUR CODE HERE FOR Q2.1"""
    model = KNN(2)
    model.fit(X,y)
    y_hat = model.predict(X_test)
    err_valid = np.mean(y_hat != y_test)
    print(err_valid)

@handle("2.2")
def q2_2():
    dataset = load_dataset("ccdebt.pkl")

    X = dataset["X"]
    y = dataset["y"]
    X_test = dataset["Xtest"]
    y_test = dataset["ytest"]

    ks = list(range(1, 30, 4))
    n, d = np.shape(X)
    print(n)
    index = np.arange(n)
    print(index)
    err_valid = np.zeros(10)
    err_mean = 0
    #print(X[(index >= 1*n/10) and (index < (1+1)*n/10)])
    print()
    """
    for i in range(len(ks)):
        err_valid = np.zeros(10)
        err_mean = 0
        for j in range(10):
            model = KNN(ks[i])
            #mask3 = (index >= j*n/10) and (index < (j+1)*n/10)
            X_test1 = X[((index.any() >= j*n/10) and (index.any() < (j+1)*n/10))==True]
            y_test1 = y[(index >= j*n/10) and (index < (j+1)*n/10)==True]
            X1 = X[(index >= j*n/10) and (index < (j+1)*n/10)==False]
            Y1 = y[(index >= j*n/10) and (index < (j+1)*n/10)==False]
            model.fit(X1, Y1)
            y_hat = model.predict(X_test1)
            err_valid[j] = np.mean(y_hat != y_test1)
        err_mean = np.mean(err_valid)
        print("k: ",ks[i], err_mean)
    """
    
    
    
    

@handle("3")
def q3():
    dataset = load_dataset("vowel.pkl")
    X = dataset["X"]
    y = dataset["y"]
    X_test = dataset["Xtest"]
    y_test = dataset["ytest"]
    print(f"n = {X.shape[0]}, d = {X.shape[1]}")

    def evaluate_model(model):
        model.fit(X, y)

        y_pred = model.predict(X)
        tr_error = np.mean(y_pred != y)

        y_pred = model.predict(X_test)
        te_error = np.mean(y_pred != y_test)
        print(f"    Training error: {tr_error:.3f}")
        print(f"    Testing error: {te_error:.3f}")

    print("Decision tree info gain")
    evaluate_model(DecisionTree(max_depth=np.inf, stump_class=DecisionStumpInfoGain))

    print("Random tree info gain")
    evaluate_model(RandomTree(max_depth=np.inf))

    print("Random forest info gain")
    evaluate_model(RandomForest(max_depth=np.inf, num_trees=50))



@handle("4")
def q4():
    X = load_dataset("clusterData.pkl")["X"]

    model = Kmeans(k=4)
    model.fit(X)
    y = model.predict(X)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap="jet")

    fname = Path("..", "figs", "kmeans_basic_rerun.png")
    plt.savefig(fname)
    print(f"Figure saved as {fname}")


@handle("4.1")
def q4_1():
    X = load_dataset("clusterData.pkl")["X"]
    
    minErr = 100000
    for i in range(50):
        for a in range(10):
            a += 1
            model = Kmeans(k=a)
            model.fit(X)
            y = model.predict(X)
            med = model.error(X, y, model.means)
            print(med)
            if (med < minErr):
                minErr = med
                plt.scatter(X[:, 0], X[:, 1], c=y, cmap="jet")

                fname = Path("..", "figs", "kmeans_basic_rerun.png")
                plt.savefig(fname)
                print(f"Figure saved as {fname}")

    print(minErr)



@handle("4.2")
def q4_2():
    X = load_dataset("clusterData.pkl")["X"]

    """YOUR CODE HERE FOR Q4.2"""

@handle("4.3")
def q4_3():
    X = load_dataset("clusterData2.pkl")["X"]

    minErr = 100000
    for i in range(50):
        model = Kmeans(k=4)
        model.fit(X)
        y = model.predict(X)
        med = model.error(X, y, model.means)
        #print(med)
        if (med < minErr):
            minErr = med
            plt.scatter(X[:, 0], X[:, 1], c=y, cmap="jet")

            fname = Path("..", "figs", "kmeans_basic_rerun.png")
            plt.savefig(fname)
            print(f"Figure saved as {fname}")

    print(minErr)


if __name__ == "__main__":
    main()
