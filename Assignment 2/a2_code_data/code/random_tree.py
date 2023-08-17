from random_stump import RandomStumpInfoGain
from decision_tree import DecisionTree
import numpy as np
from scipy import stats as st

import utils


class RandomTree(DecisionTree):
    def __init__(self, max_depth):
        DecisionTree.__init__(
            self, max_depth=max_depth, stump_class=RandomStumpInfoGain
        )

    def fit(self, X, y):
        n = X.shape[0]
        boostrap_inds = np.random.choice(n, n, replace=True)
        bootstrap_X = X[boostrap_inds]
        bootstrap_y = y[boostrap_inds]

        DecisionTree.fit(self, bootstrap_X, bootstrap_y)


class RandomForest:
    trees = []
    num_trees = 0
    def __init__(self, max_depth, num_trees):
        self.num_trees = num_trees
        for i in range(num_trees):
            self.trees.append(RandomTree(max_depth=max_depth))

    def fit(self, X, y):
        for i in range(self.num_trees):
            self.trees[i].fit(X,y)

    def predict(self, X):
        n, d = X.shape
        y = np.zeros((self.num_trees, n))

        # GET VALUES FROM MODEL
        for i in range(self.num_trees):
            j_best = self.trees[i].stump_model.j_best
            t_best = self.trees[i].stump_model.t_best
            y_hat_yes = self.trees[i].stump_model.y_hat_yes

            if j_best is None:
                # If no further splitting, return the majority label
                y[i] = y_hat_yes * np.ones(n)

            # the case with depth=1, just a single stump.
            elif self.trees[i].submodel_yes is None:
                return self.trees[i].stump_model.predict(X)

            else:
                # Recurse on both sub-models
                j = j_best
                value = t_best

                yes = X[:, j] > value
                no = X[:, j] <= value

                y[i][yes] = self.trees[i].submodel_yes.predict(X[yes])
                y[i][no] = self.trees[i].submodel_no.predict(X[no])

        #x = [[1,0],[0,1],[1,1]]
        #print(st.mode(x)[0])
        return st.mode(y)[0]