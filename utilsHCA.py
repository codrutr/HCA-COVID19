import numpy as np
import pandas as pd
import pandas.api.types as pdt
import graphicsHCA as graphics


def standardise(x):
    '''
    x - data table, expect numpy.ndarray
    '''
    means = np.mean(x, axis=0)
    stds = np.std(x, axis=0)
    Xstd = (x - means) / stds
    return Xstd


def center(x):
    '''
     x - data table, expect numpy.ndarray
     '''
    means = np.mean(x, axis=0)
    return (x - means)


def regularise(t, y=None):
    '''
    Eigenvector regularisation
    t - table of eigenvectors,
    expect either numpy.ndarray or pandas.DataFrame
    '''

    # if type(t) is pd.DataFrame:
    if isinstance(t, pd.DataFrame):
        for c in t.columns:
            minim = t[c].min()
            maxim = t[c].max()
            if abs(minim) > abs(maxim):
                t[c] = -t[c]
                if y is not None:
                    # determine column index
                    k = t.columns.get_loc(c)
                    y[:, k] = -y[:, k]
    if isinstance(t, np.ndarray):
        for i in range(np.shape(t)[1]):
            minim = np.min(t[:, i])
            maxim = np.max(t[:, i])
            if np.abs(minim) > np.abs(maxim):
                t[:, i] = -t[:, i]
    return None



def replace_na_df(t):
    '''
    replace missing values by
    mean/mode
    t - pandas.DataFrame
    '''

    for c in t.columns:
        if pdt.is_numeric_dtype(t[c]):
            if t[c].isna().any():
                avg = t[c].mean()
                t[c] = t[c].fillna(avg)
        else:
            if t[c].isna().any():
                mode = t[c].mode()
                t[c] = t[c].fillna(mode[0])
    return None


def replace_na(X):
    '''
     replace missing values by mean
     t - numpy.ndarray
     '''
    means = np.nanmean(X, axis=0)
    k_nan = np.where(np.isnan(X))
    X[k_nan] = means[k_nan[1]]
    return None


def cluster_distribution(h, k):
# TODO


# def threshold(h):
    '''
    Threshold value calculation for determining
    the maximum stability partition
    m - the maximum no. of  junctions
    '''
# TODO


# def cluster_display(g, labels, label_names, file_name):
# TODO

def color_clusters(h, k, codes):
    '''
    h - hierarchy, numpy.ndarray
    k - no. of colors
    codes - cluster codes
    '''

    colors = np.array(graphics._COLORS)
    nr_colors = len(colors)
    m = np.shape(h)[0]
    n = m + 1
    cluster_colors = np.full(shape=(2 * n * 1,),
                             fill_value="", dtype=np.chararray)
    # clusters color setting singleton
    for i in range(n):
        cluster_colors[i] = colors[codes[i] % nr_colors]
    # setting color junctions
    for i in range(m):
        k1 = int(h[i, 0])
        k2 = int(h[i, 1])
        if cluster_colors[k1] == cluster_colors[k2]:
            cluster_colors[n + i] = cluster_colors[k1]
        else:
            cluster_colors[n + i] = 'k'
    return cluster_colors
