from FairnessAwarePCA.methods import FairnessAwarePCA_MW
import pandas as pd
from sklearn.decomposition import PCA
import numpy as np

X = pd.read_csv("./data/credit/default_degree.csv", skiprows=[0])
sensitive_feature = "SEX"
sensitive_col = X[sensitive_feature]
# normalized_sensitive = (sensitive_col - 1) * (sensitive_col - 2)
# normalized_sensitive[normalized_sensitive > 0] = 1
# sensitive_col = normalized_sensitive
X = X.drop(labels=[sensitive_feature, "ID"], axis=1)
X = X - X.mean() # center data
X = X/X.std() # normalizing variance

d = 2 # parameter
T = 10
eta = 1
FairPCA = FairnessAwarePCA_MW(X, sensitive_col, d, eta, T)
P = FairPCA.fit(X)
print(np.linalg.matrix_rank(P))
#print(P_FairPCA.shape)
#lambdas, V = np.linalg.eig(P_FairPCA)
#print(P_FairPCA[abs(lambdas) == 0,:].shape)
q,r = np.linalg.qr(P.T)
#print(abs(q))
#print(abs(r))
a = P[:, :2]
print(np.linalg.matrix_rank(a))

