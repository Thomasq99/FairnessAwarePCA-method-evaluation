from FairnessAwarePCA.methods import FairnessAwarePCA_MW, preprocess_data, PostProcessing_Fairness_Aware_PCA, reconstruction_loss, FairnessAwarePCA_GD
import pandas as pd
from FairnessAwarePCA.SPEA2 import re
from sklearn.decomposition import PCA
import numpy as np
import pickle

X = pd.read_csv("./data/credit/default_degree.csv", skiprows=[0])
y = X.pop("default payment next month")
sensitive_feature = "EDUCATION"
sensitive_col = X[sensitive_feature]
normalized_sensitive = (sensitive_col - 1) * (sensitive_col - 2)
normalized_sensitive[normalized_sensitive > 0] = 1
sensitive_col = normalized_sensitive
X = X.drop(labels=[sensitive_feature, "ID"], axis=1)
# X = X - X.mean() # center data
# X = X/X.std() # normalizing variance

d = 20 # parameter
#FairPCA = PostProcessing_Fairness_Aware_PCA(sensitive_col, 2, 0.5, 30)
FairPCA = FairnessAwarePCA_GD(sensitive_col, d, 1200, 'non-pairwise')

FairPCA.fit(X)
X_normalized = X - FairPCA.mean_
X_normalized /= FairPCA.std_
print(FairPCA.components_.shape)

# FairPCA_2 = PostProcessing_Fairness_Aware_PCA(sensitive_col, 4, 0.5, 30)
# FairPCA_2.fit(X)
# print(X.mean())
# X_transformed = FairPCA_2.transform(X)
# print(X.mean())
# print('aaa')
# print(X_transformed)

#FairPCA = FairnessAwarePCA_GD(sensitive_col, d, 1000, 'non-pairwise')
#FairPCA.fit(X)
#
print(reconstruction_loss(X_normalized.to_numpy(), FairPCA.components_) / len(X))
# print(re(X.to_numpy(), X_new @ FairPCA_2.components_[:,:d].T)/ len(X_new))
# print(reconstruction_loss(X.to_numpy(), FairPCA_2.components_[:,:2])/ len(X))
# print(reconstruction_loss(X.to_numpy(), FairPCA_2.components_[:,:10])/ len(X))
# print(reconstruction_loss(X.to_numpy(), FairPCA_2.components_[:,:15])/ len(X))



# print(np.linalg.matrix_rank(FairPCA_2.components_[:,:]))

#print(FairPCA_2.components_)
# object = pd.read_pickle(r'PCA_MW_results_SEX.pickle')
# print(object['results_15'])

# lambdas, V = np.linalg.eig(FairPCA_2.components_.T)
# print(abs(lambdas))
# a = FairPCA_2.components_[abs(lambdas) > 1e-8].T
# print(reconstruction_loss(X.to_numpy(), a) / len(X))

# X = pd.read_csv("./data/credit/default_degree.csv", skiprows=[0])
# y = X.pop("default payment next month")
# sensitive_feature = "EDUCATION"
# sensitive_col = X[sensitive_feature]
# normalized_sensitive = (sensitive_col - 1) * (sensitive_col - 2)
# normalized_sensitive[normalized_sensitive > 0] = 1
# sensitive_col = normalized_sensitive
# X = X.drop(labels=[sensitive_feature, "ID"], axis=1)
#
# FairPCA_1 = PostProcessing_Fairness_Aware_PCA(sensitive_col, d, 0.5, 30)
# X_new = FairPCA_1.fit_transform(X)
# print(re(X.to_numpy(), X_new @ FairPCA_1.components_.T) / len(X))
# print(X_new.shape)
#
# print(FairPCA_1.components_.shape)

# dct = pd.read_pickle("PCA_PP_results_SEX_test.pickle")
# print(dct['results_1'])