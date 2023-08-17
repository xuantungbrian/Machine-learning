"""
Implementation of k-nearest neighbours classifier
"""

import numpy as np

import utils
from utils import euclidean_dist_squared


class KNN:
    X = None
    y = None

    def __init__(self, k):
        self.k = k

    def fit(self, X, y):
        self.X = X  # just memorize the training data
        self.y = y

    def predict(self, X_hat):
        n, d = np.shape(X_hat)
        y_hat = np.zeros(n)
        
        #calculate the distance of each object to the new ones
        distance = utils.euclidean_dist_squared(X_hat, self.X)

        #sort them
        order = np.argsort(distance, axis=1)

        #get the first k-index of the sort
        types = np.bincount(self.y)
        count = np.zeros(types.size)

        for i in range(n):
            for u in range(types.size):
                count[u] = 0
            for a in range(self.k):
                count[self.y[order[i][a]]] = count[self.y[order[i][a]]] + 1 
            y_hat[i] = np.argmax(count)
            
        return y_hat
        
        
