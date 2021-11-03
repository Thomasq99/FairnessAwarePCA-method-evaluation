from sklearn.decomposition import PCA
import numpy as np
import math
from .SPEA2 import SPEA_2_Cost, SPEA_2_fitness, SPEA_2_selection, SPEA_2_mating_pool
import cvxpy as cp
import itertools
# helper functions

def preprocess_data(data, sensitive_col):
    """preprocesses a dataset by grouping the dataset per sensitive group and centering each group
    to mean 0"""

    grouped_data = []
    for group in sensitive_col.unique():
        X_group = data[sensitive_col == group].to_numpy()
        X_group_centered = X_group - X_group.mean()
        grouped_data += [X_group_centered]
    return grouped_data

# functions for selecting a single non dominated solution

def weighted_sum_on_scales(cost, theta):
    # Weights used to select a single non-dominated solution
    a = max(cost[:,0]) - min(cost[:,0])
    b = max(cost[:,1]) - min(cost[:,1])
    py = a/(a+b)
    px = b/(a+b)

    weighted_sum = np.array([px, py]) @ cost.T # euclidean distances
    index_weight = np.argsort(weighted_sum)
    order_weight = weighted_sum[index_weight]

    # if two solutions achieve the same value, we choose the one that minimizes the RE
    if abs(order_weight[0] - order_weight[1]) < np.spacing(1):
        order_aux_ind = np.argmin([cost[index_weight[0], 0], cost[index_weight[1], 0]])
        best_cost = cost[index_weight[order_aux_ind],:]
        best_coeff = theta[index_weight[order_aux_ind]]
    else:
        best_cost = cost[index_weight[0], :]
        best_coeff = theta[index_weight[0]]
    return best_cost, best_coeff


def weighted_sum(cost, theta):
    minim = np.min(cost, axis=0)
    maxim = np.max(cost, axis=0)
    pareto_aux = np.zeros(cost.shape)
    pareto_aux[:,0] = (cost[:,0] - minim[0])/(maxim[0] - minim[0])
    pareto_aux[:,1] = (cost[:,1] - minim[1])/(maxim[1] - minim[1])

    weighted_sum = ([0.5, 0.5] @ pareto_aux.T)
    index_weight = np.argsort(weighted_sum)
    order_weight = weighted_sum[index_weight]

    # if two solutions achieve the same value, we choose the one that minimizes the RE
    if abs(order_weight[0] - order_weight[1]) < np.spacing(1):
        order_aux_ind = np.argmin([cost[index_weight[0], 0], cost[index_weight[1], 0]])
        best_cost = cost[index_weight[order_aux_ind],:]
        best_coeff = theta[index_weight[order_aux_ind]]
    else:
        best_cost = cost[index_weight[0], :]
        best_coeff = theta[index_weight[0]]
    return best_cost, best_coeff


def best_RE(cost, theta):
    smallest_RE_ind = np.argmin(cost[:,0])
    best_cost = cost[smallest_RE_ind,:]
    best_coeff = theta[smallest_RE_ind]
    return best_cost, best_coeff


def best_Fairness(cost, theta):
    smallest_Fairness_ind = np.argmin(cost[:,1])
    best_cost = cost[smallest_Fairness_ind,:]
    best_coeff = theta[smallest_Fairness_ind]
    return best_cost, best_coeff


# Actual algorithm

def PostProcessing_Fairness_Aware_PCA(X, sensitive_col, r, Pr_cross, epochs, method=None,
                                      return_cost=False):
    """ Fairness aware algorithm to select the principal components based on
     Pelegrina, G. D.; Brotto, R. D. B.; Duarte, L. T.; Attux, R. & Romano, J. M. T. (2021).
     A novel multi-objective-based approach to analyze trade-offs in Fair Principal Component Analysis.
     ArXiv preprint, arXiv:2006.06137. Available at: https://arxiv.org/abs/2006.06137"""
    num_features = len(X.columns)
    coeff = PCA().fit(X).components_.T  # each column is eigenvectors

    grouped_data = preprocess_data(X, sensitive_col)

    # BRUTE FORCE for one feature length pca:

    if r == 1:
        costs = []

        for idx_col in range(num_features):
            costs.append(SPEA_2_Cost([idx_col], coeff, grouped_data, X.to_numpy(), sensitive_col))

        costs = np.array(costs)
        include = np.array(range(num_features))

        # find all non dominated solutions
        for k in range(2):
            # in the first iteration you remove all eigenvectors that are dominated by eigenvectors with a higher eigenvalue
            idx1 = 0
            while idx1 < len(include):
                # only compare to indices higher than your own, because lower is already compared
                already_compared = np.zeros(idx1)
                # find all solutions that are dominated by the current idx
                dominated_by_current_idx = np.all((np.tile(costs[include[idx1]],
                                                           (len(include[idx1:]), 1)) < costs[
                                                       include[idx1:]]).conj().T, axis=0)
                dom_idx = np.concatenate((already_compared, dominated_by_current_idx)).astype(
                    "bool")  # True means it is dominated by another option
                include = include[
                    ~dom_idx]  # remove all eigenvectors that are dominated by the current index
                idx1 += 1
            include = np.flip(
                include)  # reverse to remove solutions dominated by solutions corresponding to lower eigenvalues

        pareto_cost = costs[include]  # costs of the non_dominated solutions
        pareto_coeff = include
        # for problems with more than one non-dominated solution:
        if len(include) != 1:
            if method == "weighted_sum_scaled":
                best_cost, best_coeff = weighted_sum_on_scales(pareto_cost, pareto_coeff)

            elif method == "weighted_sum":
                best_cost, best_coeff = weighted_sum(pareto_cost, pareto_coeff)

            elif method == "best_RE":
                best_cost, best_coeff = best_RE(pareto_cost, pareto_coeff)

            elif method == "best_Fairness":
                best_cost, best_coeff = best_Fairness(pareto_cost, pareto_coeff)

            elif method == None:
                return pareto_cost, pareto_coeff

        else:  # only one dominated solution
            pareto_cost = pareto_cost[0]  # remove outer list
            best_cost = pareto_cost
            best_coeff = pareto_coeff

    else:
        # FOR MORE THAN ONE FEATURE USE MOFPCA
        # MO parameters
        pop_size = min(100, round(math.comb(num_features, r) / 2))
        ext_set_size = round(pop_size / 2)
        ParK = round(math.sqrt(pop_size + ext_set_size))  # SPEA 2 parameter
        cross_max = round(Pr_cross * pop_size / 2) * 2  # Max number of crossover
        mutation_max = pop_size - cross_max  # max number of mutation

        # initialization
        # initialize population
        population = []
        i = 1
        while i <= pop_size:  # population with different elements
            aux_pop = tuple(np.sort(np.random.permutation(num_features)[:r]))
            pop_temp = population.copy()
            pop_temp.append(aux_pop)
            if len(set(pop_temp)) == i:
                population = pop_temp
                i += 1
        population = np.array(population)

        ext_set = []
        ext_set_cost = []

        for iteration in range(epochs):
            population_external = population.astype(int)

            if len(ext_set) != 0:
                population_external = np.append(population_external, ext_set, axis=0)
            population_external_size = population_external.shape[0]
            pop_ext_cost = np.zeros((pop_size, 2))
            pop_ext_cost[0, :] = SPEA_2_Cost(population_external[0, :], coeff, grouped_data,
                                             X.to_numpy(), sensitive_col)

            for i in range(1, pop_size):
                if np.all(population_external[i, :] == population_external[i - 1, :]):
                    pop_ext_cost[i, :] = pop_ext_cost[i - 1, :]
                else:
                    pop_ext_cost[i, :] = SPEA_2_Cost(population_external[i, :], coeff,
                                                     grouped_data, X.to_numpy(), sensitive_col)

            if len(ext_set_cost) != 0:
                pop_ext_cost = np.append(pop_ext_cost, ext_set_cost, axis=0)

            # calculate fitness
            pop_ext_fit = SPEA_2_fitness(population_external_size, pop_ext_cost, ParK)

            # selection
            ext_set, ext_set_cost, ext_set_fit = SPEA_2_selection(ext_set_size,
                                                                  population_external_size,
                                                                  pop_ext_fit, population_external,
                                                                  pop_ext_cost)

            # Stop criteria (seems redundant)
            if iteration >= epochs:
                break

            population = SPEA_2_mating_pool(mutation_max, cross_max, ext_set, ext_set_size,
                                            pop_size, ext_set_fit, iteration, epochs, r,
                                            num_features)
            population = population[population[:, 0].argsort()]

            if (iteration + 1) % 5 == 0:
                print("Finished iteration ", iteration + 1)

        # Updating non-dominated solutions
        pareto_cost, pareto_cost_idx = np.unique(ext_set_cost, return_index=True, axis=0)
        pareto_coeff = ext_set[pareto_cost_idx, :]

        # selecting a single non dominated solution
        # for problems with more than one non-dominated solution:
        if pareto_cost.shape[0] != 1:
            if method == "weighted_sum_scaled":
                best_cost, best_coeff = weighted_sum_on_scales(pareto_cost, pareto_coeff)

            elif method == "weighted_sum":
                best_cost, best_coeff = weighted_sum(pareto_cost, pareto_coeff)

            elif method == "best_RE":
                best_cost, best_coeff = best_RE(pareto_cost, pareto_coeff)

            elif method == "best_Fairness":
                best_cost, best_coeff = best_Fairness(pareto_cost, pareto_coeff)

            elif method == None:
                if return_cost:
                    return pareto_cost, pareto_coeff
                else:
                    return pareto_coeff

        else:  # only one dominated solution
            pareto_cost = pareto_cost[0]  # remove outer list
            best_cost = pareto_cost
            best_coeff = pareto_coeff

    if return_cost:
        return best_cost, best_coeff
    else:
        return best_coeff


def loss(Y, Z, Yhat):
    return np.linalg.norm(Y-Z, ord='fro')**2 - np.linalg.norm(Y-Yhat, ord='fro')**2


def find_optimal1(grouped_data, r):
    optimals = []
    for data in grouped_data:
        pca = PCA(n_components=r)
        coeff = pca.fit(data).components_.T  # this is the projection matrix
        P = coeff @ coeff.T
        optimals += [data @ P]
    return optimals


def oracle(grouped_data, weights_vec, covariances, constants, d):
    pca = PCA(d)
    weighted_grouped_data = [np.sqrt((1/grouped_data[i].shape[0])*weights_vec[i])*grouped_data[i] for i in range(len(grouped_data))]
    weighted_data = np.concatenate(weighted_grouped_data, axis=0)
    coeff_P_o = pca.fit(weighted_data).components_.T
    P_o = coeff_P_o @ coeff_P_o.T
    z_vec = [(1/grouped_data[i].shape[0])*(constants[i] - sum(sum(covariances[i]*P_o))) for i in range(len(grouped_data))]
    return P_o, z_vec


def FairnessAwareMW(data, sensitive_col, d, eta, T, return_optimals):
    grouped_data = preprocess_data(data, sensitive_col)

    # algorithm
    no_of_cols = data.shape[1]

    # compute covariances
    covariances = [matrix.T @ matrix for matrix in grouped_data]
    optimals = find_optimal1(grouped_data, d)
    constants = [np.linalg.norm(optimal, "fro") ** 2 for optimal in optimals]

    # start MW
    # uniform weights

    weights = [1 / len(grouped_data)] * len(grouped_data)
    P = np.zeros((no_of_cols, no_of_cols))

    for t in range(1, T + 1):
        P_temp, z_vec = oracle(grouped_data, weights, covariances, constants, d)
        weights_star = [weights[i] * np.exp(eta * z_vec[i]) for i in range(len(weights))]

        # renormalize
        weights = [weight / (sum(weights_star)) for weight in weights_star]

        P += P_temp

        P_average = (1 / t) * P
        avg_loss = [loss(grouped_data[i], grouped_data[i] @ P_average, optimals[i]) /
                    grouped_data[i].shape[0] for i in range(len(grouped_data))]

    P = (1 / T) * P
    z_vec = [(1 / grouped_data[i].shape[0]) * (constants[i] - sum(sum(covariances[i] * P))) for i
             in range(len(grouped_data))]
    z = max(z_vec)

    # if last iterate is preferred to average:
    P_last = P_temp

    # calculate loss
    z_vec_last = [
        (1 / grouped_data[i].shape[0]) * (constants[i] - sum(sum(covariances[i] * P_last))) for i
        in range(len(grouped_data))]

    z_last = max(z_vec_last);

    if z_last < z:
        P = P_last
    P = np.identity(P.shape[0]) - sqrtm(np.identity(P.shape[0]) - P)

    if return_optimals:
        return P, optimals

    else:
        return P


def find_optimal2(grouped_data, r):
    U = []
    for X_group in grouped_data:
        pca = PCA(n_components=r)
        U_group = pca.fit(X_group).components_.T # this is the projection matrix
        U += [U_group]
    return U


def reconstruction_loss(X_i, U):
    return np.linalg.norm((X_i - (X_i @ U @ U.T)), ord='fro')**2


def Eps_i(X_i, U, U_i_optimal):
    return reconstruction_loss(X_i, U) - reconstruction_loss(X_i, U_i_optimal)

def phi(z):
    return 1/2*z**2


def solve_QP(gradients):
    lambda_ = cp.Variable(len(gradients))
    constraints = [cp.sum(lambda_) == 1, lambda_ >= 0]
    exp = 0
    for i in range(len(gradients)):
        exp += lambda_[i] * gradients[i]
    prob = cp.Problem(cp.Minimize((1/2) * (cp.norm(exp, 'fro')**2)), constraints)
    prob.solve(solver=cp.ECOS)
    return lambda_.value


def objective_non_pairwise(X, U, grouped_data, optimals):
    objectives = [reconstruction_loss(X, U)]
    for idx, X_group in enumerate(grouped_data):
        objectives.append(phi(Eps_i(X_group, U, optimals[idx])))
    return objectives


def objective_pairwise(X, U, grouped_data, optimals):
    objectives = [reconstruction_loss(X, U)]

    # get all combinations
    group_combinations = itertools.combinations(range(len(grouped_data)), 2)
    for comb in group_combinations:
        group1, group2 = comb[0], comb[1]
        X_1 = grouped_data[group1]
        X_2 = grouped_data[group2]

        delta_Eps = Eps_i(X_1, U, optimals[group1]) - Eps_i(X_2, U, optimals[group2])
        objectives.append(phi(delta_Eps))
    return objectives


def FairnessAwarePCA_GD(X, grouped_data, r, num_of_iterations, loss='non-pairwise', return_optimal=False):
    d = len(X.columns)
    optimals = find_optimal2(grouped_data, r)  # find optimal rank r subspace for each group
    X_arr = X.to_numpy()
    U, _ = np.linalg.qr(np.random.randn(d,
                                        r))  # randomly generate initial projection matrix, with orthonormal columns

    # create objective vector (without regularization term for now)
    if loss == 'non-pairwise':
        objectives = objective_non_pairwise(X_arr, U, grouped_data, optimals)
    elif loss == 'pairwise':
        objectives = objective_pairwise(X_arr, U, grouped_data, optimals)
    else:
        raise ValueError("loss must be pairwise or non-pairwise")

    print(objectives)

    for t in range(num_of_iterations):

        lr = 1 / np.sqrt(t + 1)

        # calculate the gradient matrices for each objective
        G_0 = -2 * X_arr.T @ X_arr @ U
        G_0 /= np.linalg.norm(G_0, ord=None)
        gradients = [G_0]

        for X_group in grouped_data:
            G_i = -2 * X_group.T @ X_group @ U
            gradients += [G_i / np.linalg.norm(G_i, ord=None)]

        # find descent direction
        lambda_ = solve_QP(gradients)
        D = 0
        for i in range(len(lambda_)):
            D += lambda_[i] * gradients[i]
        D = -1 * D

        if np.allclose(D, np.zeros(D.shape)):
            print("converged")
            if return_optimal:
                return U, optimals
            else:
                return U
        else:
            U, _ = np.linalg.qr(
                U + lr * D)  # apply Gram-Schmidt procedure to make columns orthornormal
            if (t + 1) % 100 == 0:
                print("In iteration number ", t + 1, " reconstruction loss is:",
                      reconstruction_loss(X_arr, U) / X_arr.shape[0])
    if return_optimal:
        return U, optimals
    else:
        return U