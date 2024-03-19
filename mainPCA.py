import pandas as pd
import functions as fun
import pca.PCA as pca
import graphicsPCA as g

#reading from the csv file
print('Showing data from DataSet.csv')
table = pd.read_csv('dataIN//DataSet.csv', index_col= 0)
print(table)

# selecting the relevant variable based on the purpose of the analysis
print('\n\nSELECTING RELEVANT VARIABLES BASED ON THE PURPOSE OF PCA ANALYSIS:')
vars = table.columns.values[1:]
print(vars, type(vars))

# creating a list of variables
print('\n\nCREATING THE LIST OF VARIABLES:')
varName = list(table.columns)[1:]
#print(list)

# creating a list of observations
print('\n\nCreating a list of observations:')
obs = table.index.values
print(obs, type(obs))
print('\n\n')

X = table[vars].values
print(X, type(X))

# standardise the initial variables
print('\n\nStandardising the initial variables:')
Xstd = fun.standardise(X)

#saving Xstd into a CSV file
print('\n\nsaving Xstd into a csv file process')
Xstd_df = pd.DataFrame(data=Xstd, index=obs, columns=vars)
print(Xstd_df)
Xstd_df.to_csv('./dataOUT/Xstd.csv')

#pca model
pcaModel = pca.PCA(Xstd)

#extracting the eigenvalues
print('\n\nExtracting the eigenvalues')
alpha = pcaModel.getAlpha()
print(alpha)

#First graphic
g.principalComponents(eigenvalues=alpha)
g.show()

# extracting the principal components
# print('\n\nExtracting the principal components')
prinComp = pcaModel.getPrinComp()
# converting the numpy.ndarray into a pandas.DataFrame
components = ['C'+str(j+1) for j in range(prinComp.shape[1])]
prinComp_df = pd.DataFrame(data=prinComp, index=obs, columns=components)

# saving the principal components into a CSV
prinComp_df.to_csv('./dataOUT/PrincComp.csv')

# extracting the factor loadings
factorLoadings = pcaModel.getFactorLoadings()
factorLoadings_df = pd.DataFrame(data=factorLoadings, index=vars, columns=components)

#saving the factor loadings data into a CSV file
factorLoadings_df.to_csv('./dataOUT/FactorLoadings.csv')

# creating the correlogram of factor loadings
g.correlogram(matrix=factorLoadings_df, title='Correlogram showing the factor analysis')
g.show()

# extracting the scores
scores = pcaModel.getScores()
#saving the quality of point representations into a CSV file
scores_df = pd.DataFrame(data=scores, index=obs, columns=components)
scores_df.to_csv('./dataOUT/Scores.csv')
g.intensity_map(matrix=scores_df, title='Correlogram showing standardized principal components of the scores')
g.show()

# extracting the quality of points representations
qualObs = pcaModel.getQualObs()
qualObs_df = pd.DataFrame(data=qualObs, index=obs, columns=components)
qualObs_df.to_csv('./dataOUT/ObservationQuality.csv')
g.intensity_map(matrix=qualObs_df, title='Quality of points representation')
g.show()

# extracting the contribution of observations to the axes' variance
contribObs = pcaModel.getContribObs()
contribObs_df = pd.DataFrame(data=contribObs, index=obs, columns=components)
g.intensity_map(matrix=contribObs_df, title="Correlogram showing the contribution of observations to the axes of variance")
g.show()

# extracting the commonalities
common = pcaModel.getCommon()
# saving the commonalities into a CSV file
common_df = pd.DataFrame(data=common, index=vars, columns=components)
common_df.to_csv('./dataOUT/Commonalities.csv')
g.intensity_map(matrix=common_df, title='Commonalities')

g.show()






