from FairnessAwarePCA.methods import FairnessAwarePCA_MW, FairnessAwarePCA_GD, PostProcessing_Fairness_Aware_PCA
import pandas as pd
import numpy as np

X = pd.read_csv("./data/credit/default_degree.csv", skiprows=[0])
sensitive_feature = "EDUCATION"
sensitive_col = X[sensitive_feature]
normalized_sensitive = (sensitive_col - 1) * (sensitive_col - 2)
normalized_sensitive[normalized_sensitive > 0] = 1
sensitive_col = normalized_sensitive
X = X.drop(labels=[sensitive_feature, "ID"], axis=1)

d = 2 # parameter
T = 10
eta = 1
PCA1 = FairnessAwarePCA_MW(sensitive_col, d , eta, T)
print(PCA1.fit_transform(X).shape)

PCA2 = FairnessAwarePCA_GD(sensitive_col, 2, 100, "non-pairwise")
PCA2.fit(X)
X_new = PCA2.transform(X)
print(X_new.shape)
print(PCA2.fit_transform(X).shape)

PCA3 = PostProcessing_Fairness_Aware_PCA(sensitive_col, 2, 0.5, 10,"weighted_sum_scaled")
PCA3.fit(X)
X_new = PCA3.transform(X)
print(X_new.shape)
print(PCA3.fit_transform(X).shape)
