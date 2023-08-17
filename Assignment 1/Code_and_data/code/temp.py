class DecisionStumpErrorRate:
    y_hat_yes = None
    y_hat_no = None
    t_best = None

    def fit(self, X, y):
        #Find the number of features and objects
        n, d = X.shape
        minError = n
        self.y_hat_yes = []
        self.y_hat_no = []
        self.t_best = []
        Y = [X] 
        x = [y] 
        y_pred = [None] * (pow(2,d)-1)
        for j in range(pow(2,d)-1):
            if (j>=len(x)):
                break
            if np.unique(x[j]).size <= 1:
                if (x[j] is None):
                    pass
                else:
                    self.y_hat_yes[j] = x[j][0]
                    self.y_hat_no[j] = x[j][0]
                continue
            minError = n
            a = 0
            if (j >= pow(2,(a+1))-1):
                a = a + 1

            n1, d1 = Y[j].shape
            for i in range(n1):
                t = np.round(Y[j][i, a])

                # Find most likely class for each split
                larger = np.round(Y[j][:, a]) > t
                y_yes_mode = utils.mode(x[j][larger])
                y_no_mode = utils.mode(x[j][~larger])  # ~ is "logical not"

                # Make predictions
                y_pred[j] = y_yes_mode * np.ones(n1)
                y_pred[j][np.round(Y[j][:, a]) <= t] = y_no_mode

                # Compute error
                errors = np.sum(y_pred[j] != x[j])

                # Compare to minimum error so far
                if errors < minError:
                    # This is the lowest error, store this value
                    minError = errors
                    min_t = t
                    min_yes = y_yes_mode
                    min_no = y_no_mode
            self.t_best.append(min_t)
            self.y_hat_yes.append(min_yes)
            self.y_hat_no.append(min_no)
            if (pow(2, d)-2>=2*j+2):
                larger3 = np.round(Y[j][:, a]) > self.t_best[j]
                Y.append(Y[j][larger3])
                x.append(x[j][larger3])
                Y.append(Y[j][~larger3])
                x.append(x[j][~larger3])

    def predict(self, X):
        n, d = X.shape
        X = np.round(X)
        y_hat = np.zeros(n)
        for i in range(n):
            y_hat[i] = DecisionStumpErrorRate.determine(i, 0, 0, X, self)
        return y_hat

    def determine(index, heap_num, depth, X, self):
        if (pow(2, depth+1)-1 == len(self.t_best)):
            if X[index, depth] > self.t_best[heap_num]:
                return self.y_hat_yes[heap_num]
            else:
                return self.y_hat_no[heap_num]
        elif X[index, depth] > self.t_best[heap_num]:
            return DecisionStumpErrorRate.determine(index, heap_num*2+1, depth+1, X, self)
        else:
            return DecisionStumpErrorRate.determine(index, heap_num*2+2, depth+1, X, self)

    def predict(self, X):
        n, d = X.shape

        y_hat = np.zeros(n)

        for i in range(n):
            if X[i, 0] > -80.305106:
                y_hat[i] = 0
            else:
                if X[i, 1] > 37.669007:
                    y_hat[i] = 0
                else:
                    y_hat[i] = 1
        return y_hat