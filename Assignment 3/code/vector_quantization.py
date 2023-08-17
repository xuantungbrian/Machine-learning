import numpy as np
from utils import load_image
from sklearn.cluster import KMeans

# Vector Quantization
class VectorQuantization:

    @classmethod
    def quantizeImage(cls, filename, b):
        I = load_image(filename)
        nRows, nCols, h = I.shape
        d2_I = I.reshape((nRows*nCols, h))
        kmeans = KMeans(n_clusters=2**b, random_state=0, n_init="auto").fit(d2_I)
        y = kmeans.labels_
        W = kmeans.cluster_centers_
        return y, W, nRows, nCols

    @classmethod
    def deQuantizeImage(cls, y, W, nRows, nCols):
        I = np.zeros((nRows, nCols, 3))
        for a in range(nRows):
            for x in range(nCols):
                for h in range(3):
                    I[a][x][h] = W[y[a*nCols+x]][h]
        return I

        