from FairnessAwarePCA.methods import FairnessAwarePCA_MW, FairnessAwarePCA_GD, \
    PostProcessing_Fairness_Aware_PCA, preprocess_data
from FairnessAwarePCA.SPEA2 import re
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
import pandas as pd
import numpy as np
import pickle
from sklearn.pipeline import Pipeline

X = pd.read_csv("./data/credit/default_degree.csv", skiprows=[0])
y = X.pop("default payment next month")
sensitive_feature = "SEX"
sensitive_col = X[sensitive_feature] - 1
X = X.drop(labels=[sensitive_feature, "ID"], axis=1)
columns = X.columns


RE_pp_PCA = []
RE_FairPCA = []
RE_EfficientFairPCA_non_pairwise = []
RE_EfficientFairPCA_pairwise = []
RE_PCA = []
results = {}

for d in range(1, 21):
    print('Started experiment for d= ' + str(d))

    # parameters are made sure to make the algorithm converge
    FairPCA = FairnessAwarePCA_MW(sensitive_col, d, 1, 10)
    EfficientFairPCA_non_pairwise = FairnessAwarePCA_GD(sensitive_col, d, 2000, 'non-pairwise')
    EfficientFairPCA_pairwise = FairnessAwarePCA_GD(sensitive_col, d, 2000, 'pairwise')
    postprocessingPCA = PostProcessing_Fairness_Aware_PCA(sensitive_col, d, 0.5, 30)
    pca = PCA(d)#Pipeline([('scaler', StandardScaler()), ('pca', PCA(d))])

    algorithms = [FairPCA, EfficientFairPCA_non_pairwise, EfficientFairPCA_pairwise, postprocessingPCA]
    names = ['FairPCA', 'EfficientFairPCA_non_pairwise', 'EfficientFairPCA_pairwise', 'PostProcessingPCA']

    for name, algorithm in zip(names, algorithms):
        RE_lst_test = []
        RE_group0_lst_test = []
        RE_group1_lst_test = []
        RE_lst_train = []
        RE_group0_lst_train = []
        RE_group1_lst_train = []

        skf = StratifiedKFold(n_splits=10)
        for train_index, test_index in skf.split(X, sensitive_col):
            X_train, X_test = X.loc[train_index, :], X.loc[test_index, :]

            # fit and transform data
            algorithm.fit(X_train)

            # performance on test data:
            X_test_transformed = algorithm.transform(X_test)

            # normalize data for computing RE using only the train data to simulate real life scenario
            X_test_normalized = X_test.copy()
            X_test_normalized -= X_train.mean()
            X_test_normalized /= X_train.std()

            X_test_transformed = X_test_transformed @ algorithm.components_[:, :d].T

            # calculate overall RE
            RE = re(X_test_normalized.to_numpy(), X_test_transformed) / len(X_test)
            RE_lst_test.append(RE)

            # performance on train data:
            X_train_transformed = algorithm.transform(X_train)

            # normalize data for computing RE using only the train data to simulate real life scenario
            X_train_normalized = X_train.copy()
            X_train_normalized -= X_train.mean()
            X_train_normalized /= X_train.std()
            X_train_transformed = X_train_transformed @ algorithm.components_[:, :d].T

            # calculate overall RE
            RE = re(X_train_normalized.to_numpy(), X_train_transformed) / len(X_train)
            RE_lst_train.append(RE)

            # calculate group RE for train data
            groups = preprocess_data(X_train, sensitive_col)
            for idx, group_data in enumerate(groups):

                if isinstance(group_data, np.ndarray):
                    group_data = pd.DataFrame(group_data, columns=columns)

                group_data_transformed = algorithm.transform(group_data)

                # TODO check if I need to use X.mean or group_data.mean
                group_data_normalized = group_data.copy()
                group_data_normalized -= X.mean()
                group_data_normalized /= X.std()
                group_data_transformed = group_data_transformed @ algorithm.components_[:, :d].T

                RE_group = re(group_data_normalized.to_numpy(), group_data_transformed) / len(group_data)

                if idx == 0:
                    RE_group0_lst_train.append(RE_group)
                elif idx == 1:
                    RE_group1_lst_train.append(RE_group)

            # calculate group RE for test data
            groups = preprocess_data(X_test, sensitive_col)
            for idx, group_data in enumerate(groups):

                if isinstance(group_data, np.ndarray):
                    group_data = pd.DataFrame(group_data, columns=columns)

                group_data_transformed = algorithm.transform(group_data)

                # TODO check if I need to use X.mean or group_data.mean
                group_data_normalized = group_data.copy()
                group_data_normalized -= X.mean()
                group_data_normalized /= X.std()

                group_data_transformed = group_data_transformed @ algorithm.components_[:, :d].T

                RE_group = re(group_data_normalized.to_numpy(), group_data_transformed) / len(group_data)

                if idx == 0:
                    RE_group0_lst_test.append(RE_group)
                elif idx == 1:
                    RE_group1_lst_test.append(RE_group)

        average_RE_overall_train = np.array(RE_lst_train).mean()
        average_RE_0_train = np.array(RE_group0_lst_train).mean()
        average_RE_1_train = np.array(RE_group1_lst_train).mean()

        std_RE_overall_train = np.array(RE_lst_train).std()
        std_RE_0_train = np.array(RE_group0_lst_train).std()
        std_RE_1_train = np.array(RE_group1_lst_train).std()

        average_RE_overall_test = np.array(RE_lst_test).mean()
        average_RE_0_test = np.array(RE_group0_lst_test).mean()
        average_RE_1_test = np.array(RE_group1_lst_test).mean()

        std_RE_overall_test = np.array(RE_lst_test).std()
        std_RE_0_test = np.array(RE_group0_lst_test).std()
        std_RE_1_test = np.array(RE_group1_lst_test).std()

        # train data
        key_overall = name + '_train_overall'
        if key_overall in results:
            results[key_overall] += [average_RE_overall_train]
        else:
            results[key_overall] = [average_RE_overall_train]

        key_0 = name + '_train_Male'
        if key_0 in results:
            results[key_0] += [average_RE_0_train]
        else:
            results[key_0] = [average_RE_0_train]

        key_1 = name + '_train_Female'
        if key_1 in results:
            results[key_1] += [average_RE_1_train]
        else:
            results[key_1] = [average_RE_1_train]

        key_overall = name + '_train_overall_std'
        if key_overall in results:
            results[key_overall] += [std_RE_overall_train]
        else:
            results[key_overall] = [std_RE_overall_train]

        key_0 = name + '_train_Male_std'
        if key_0 in results:
            results[key_0] += [std_RE_0_train]
        else:
            results[key_0] = [std_RE_0_train]

        key_1 = name + '_train_Female_std'
        if key_1 in results:
            results[key_1] += [std_RE_1_train]
        else:
            results[key_1] = [std_RE_1_train]

        # test data
        key_overall = name + '_test_overall'
        if key_overall in results:
            results[key_overall] += [average_RE_overall_test]
        else:
            results[key_overall] = [average_RE_overall_test]

        key_0 = name + '_test_Male'
        if key_0 in results:
            results[key_0] += [average_RE_0_test]
        else:
            results[key_0] = [average_RE_0_test]

        key_1 = name + '_test_Female'
        if key_1 in results:
            results[key_1] += [average_RE_1_test]
        else:
            results[key_1] = [average_RE_1_test]

        key_overall = name + '_test_overall_std'
        if key_overall in results:
            results[key_overall] += [std_RE_overall_test]
        else:
            results[key_overall] = [std_RE_overall_test]

        key_0 = name + '_test_Male_std'
        if key_0 in results:
            results[key_0] += [std_RE_0_test]
        else:
            results[key_0] = [std_RE_0_test]

        key_1 = name + '_test_Female_std'
        if key_1 in results:
            results[key_1] += [std_RE_1_test]
        else:
            results[key_1] = [std_RE_1_test]

f = open("experiment_RE_CV.pickle", "wb")
pickle.dump(results, f)
f.close()
