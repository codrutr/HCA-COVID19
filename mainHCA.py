import pandas as pd

import utilsHCA as utils

import scipy.cluster.hierarchy as hclust
import scipy.spatial.distance as hdist

import matplotlib as mpl

# try:
fileName = './dataIN/DataSet.csv'

mpl.rcParams['figure.max_open_warning'] = 50
print(mpl.rcParams['figure.max_open_warning'])
print('\n')

table = pd.read_csv(fileName, index_col=1)
utils.replace_na_df(table)
labelVars = list(table)
columnLabel = 'Label'

print("DISPLAYING ALL THE COLUMN 'Label' VALUES: ")
print(table[columnLabel].values)

print('\nTABLE INDEX:')
print(table.index)
print('\nTABLE INDEX NAME:')
print(table.index.name)

obs = table[columnLabel].values
# print(obs)

# standardising
vars = labelVars[2:]
X = table[vars].values
Xstd = utils.standardise(X)
Xstd_df = pd.DataFrame(data=Xstd, index=obs, columns=vars)
print('\nSTANDARDISING THE VARIABLES:')
print(Xstd_df)

# Creating hierarchical instances
methods = list(hclust._LINKAGE_METHODS)
metrics = hdist._METRICS_NAMES
print('\n\nMethods: ', methods)
print('\nMetrics: ', metrics)

method = methods[5]  # ward
print('\nMethod 6: ', method)
distance = metrics[7]  # euclidean
print('\nMetric 8: ', distance)


if (method == 'ward' or method == 'centroid' or
        method == 'median' or method == 'weighted'):
    distance = 'euclidean'
else:
    distance = metrics[7]

# except Exception as ex:
# print("Error!", ex.with_traceback(), sep="\n")