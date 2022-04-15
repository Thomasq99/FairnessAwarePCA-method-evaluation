from FairnessAwarePCA.methods import FairnessAwarePCA_MW, preprocess_data, PostProcessing_Fairness_Aware_PCA, reconstruction_loss, FairnessAwarePCA_GD
import pandas as pd
from FairnessAwarePCA.SPEA2 import re
import numpy as np
import pickle
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score, accuracy_score, recall_score
import sklearn
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from scipy.stats import loguniform




X = pd.read_csv("./data/credit/default_degree.csv", skiprows=[0])
print(X.shape)
y = X.pop("default payment next month")
sensitive_feature = "SEX"
sensitive_col = X[sensitive_feature]
# normalized_sensitive = (sensitive_col - 1) * (sensitive_col - 2)
# normalized_sensitive[normalized_sensitive > 0] = 1
# sensitive_col = normalized_sensitive
X = X.drop(labels=[sensitive_feature, "ID"], axis=1)
# X = X - X.mean() # center data
# X = X/X.std() # normalizing variance

pca = PCA(10)


X_normalized = X.copy()
X_normalized -= X.mean()
X_normalized /= X.std()

pca.fit(X_normalized)

X_transformed = pca.transform(X_normalized)
print(pca.inverse_transform(X_transformed)[0:10, 0:2])
print(X_normalized.iloc[0:10, 0:2])
print(len(X_normalized))
RE = re(X_normalized.to_numpy(), pca.inverse_transform(X_transformed)) / len(X_normalized)
print(RE)

print(y.value_counts())
# pipe = Pipeline([('PCA', PCA(10)), ('svc', sklearn.svm.SVC(class_weight='balanced'))])
# p_grid_svm = {"svc__C": loguniform(2 ** -5, 2 ** 10), "svc__gamma": loguniform(2 ** -15, 2)}
#
# inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=1)
#
# clf = RandomizedSearchCV(estimator=pipe, param_distributions=p_grid_svm, cv=inner_cv, scoring='roc_auc', verbose=2,
#                              n_iter=75, n_jobs=-1, refit=True, error_score='raise')
#
# # alg = sklearn.svm.SVC(C=0.04, gamma=0.06)
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, stratify=y)
# search = clf.fit(X_train, y_train)
# print(search.best_params_)
# print(search.best_score_)
# yhat = clf.predict(X_test)
# print(accuracy_score(y_test, yhat))
# print(roc_auc_score(y_test, yhat))
# print(recall_score(y_test, yhat))
# print(pd.Series(yhat).value_counts())
# print(pd.Series(y_test).value_counts())
# print(y.value_counts())


