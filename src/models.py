import logging
import numpy as np
import sklearn
from scipy.spatial.distance import cosine, braycurtis, canberra, cityblock

def minmax(a, b):
    return np.minimum(a, b).sum() / np.maximum(a, b).sum()

def similarities(a, b):
    # NOTE using all features of a and b
    return (minmax(a, b),
            cosine(a, b),
            braycurtis(a, b),
            canberra(a, b),
            cityblock(a, b))

class Models:
    def __init__(self):
        logging.debug('Loading pickled models...')
        self.svm = sklearn.externals.joblib.load('../data/models/svm.pk')
        # TODO other models?
        logging.debug('Done.')

    def classify(self, a, b):
        # NOTE: a and b are expected to have ndim = 1!
        # 1st model - svm similarities:
        s = np.array(similarities(a, b)).reshape(1, -1)
        y_pred_svm = self.svm.predict(s)
        # TODO
