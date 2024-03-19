import numpy as np


def standardise(X):  # the assumption is that X is numpy.ndarray
    means = np.mean(a=X, axis=0)  # means on the columns
    stds = np.std(a=X, axis=0)
    return (X - means) / stds
