from FairnessAwarePCA.methods import FairnessAwarePCA_MW, reconstruction_loss, FairnessAwarePCA_GD, \
    PostProcessing_Fairness_Aware_PCA, preprocess_data
from FairnessAwarePCA.SPEA2 import re
import pandas as pd
import numpy as np
import sklearn
from scipy.stats import loguniform
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score
import pickle

X = pd.read_csv("./data/credit/default_degree.csv", skiprows=[0])
y = X.pop("default payment next month")
sensitive_feature = "SEX"
sensitive_col = X[sensitive_feature] - 1
# normalized_sensitive = (sensitive_col - 1) * (sensitive_col - 2)
# normalized_sensitive[normalized_sensitive > 0] = 1
# sensitive_col = normalized_sensitive
X = X.drop(labels=[sensitive_feature, "ID"], axis=1)
columns = X.columns
# defining the parameters
T = 10
eta = 1

SVC = sklearn.svm.SVC(kernel='rbf')

p_grid_svm = {"svc__C": loguniform(2 ** -5, 2 ** 10), "svc__gamma": loguniform(2 ** -15, 2)}

# TODO need the folds differ per d
inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=1)
outer_cv = StratifiedKFold(n_splits=6, shuffle=True, random_state=12)

results_dct = {}
for d in range(1, 21):
    pipe = Pipeline([('PCA', FairnessAwarePCA_MW(sensitive_col, d, eta, T)), ('svc', SVC)])
    #pipe = Pipeline([('PCA', PostProcessing_Fairness_Aware_PCA(sensitive_col, d, 0.5, 30)), ('svc', SVC)])
    clf = RandomizedSearchCV(estimator=pipe, param_distributions=p_grid_svm, cv=inner_cv, scoring='roc_auc', verbose=2,
                             n_iter=75, n_jobs=-1, refit=True, error_score='raise')

    roc_auc_lst = []
    RE_lst = []
    RE_dct = {}
    AUC_dct = {}
    iter = 0
    for train_ix, test_ix in outer_cv.split(X, y):
        iter += 1
        # select rows
        train_X, test_X = X.loc[train_ix, :], X.loc[test_ix, :]
        train_y, test_y = y.loc[train_ix], y.loc[test_ix]


        # inner CV
        search = clf.fit(train_X, train_y)
        best_model = search.best_estimator_

        # compute roc_auc
        # evaluate model on the hold out dataset
        yhat = best_model.predict(test_X)
        # evaluate the model
        roc_auc = roc_auc_score(test_y, yhat)
        roc_auc_lst.append(roc_auc)

        pca = best_model['PCA']

        # compute fairness score for entire dataset:

        # TODO determine which one to use
        #RE = reconstruction_loss(test_X.to_numpy(), pca.components_) / len(test_X)
        test_X_transformed = pca.transform(test_X)

        # normalize test_x
        test_X_normalized = test_X.copy()
        test_X_normalized -= pca.mean_
        if pca.normalized:
            test_X_normalized /= pca.std_

        RE = re(test_X_normalized.to_numpy(), test_X_transformed @ pca.components_[:,:d].T) / len(test_X)
        RE_lst.append(RE)

        # compute fairness score per group:
        groups = preprocess_data(test_X, sensitive_col)
        groups_y = preprocess_data(test_y, sensitive_col)

        for idx, group_data in enumerate(groups):

            if isinstance(group_data, np.ndarray):
                group_data = pd.DataFrame(group_data, columns=columns)

            group_data_transformed = pca.transform(group_data)

            # normalize group_data
            group_data -= pca.mean_
            if pca.normalized:
                group_data /= pca.std_


            re_group = re(group_data.to_numpy(), group_data_transformed @ pca.components_[:,:d].T) / len(group_data)
            yhat_group = best_model.predict(group_data)
            y_group = groups_y[idx]

            roc_auc_group = roc_auc_score(y_group, yhat_group)

            if str(idx) in RE_dct:
                RE_dct[str(idx)] += [re_group]
            else:
                RE_dct[str(idx)] = [re_group]

            if str(idx) in AUC_dct:
                AUC_dct[str(idx)] += [roc_auc_group]
            else:
                AUC_dct[str(idx)] = [roc_auc_group]
        print("\n FINISHED ITERATION {} FOR d = {} \n".format(iter, d))

    df = pd.DataFrame()
    df['AUC_overall'] = roc_auc_lst
    df['RE_overall'] = RE_lst

    for group in sensitive_col.unique():
        df['AUC_group_{}'.format(str(group))] = AUC_dct[str(group)]
        df['RE_group_{}'.format(str(group))] = RE_dct[str(group)]

    results_dct["results_{}".format(d)] = df
    print("finished d =  {}".format(d))

    with open('PCA_MW_results_SEX.pickle', 'wb') as handle:
        pickle.dump(results_dct, handle, protocol=pickle.HIGHEST_PROTOCOL)
