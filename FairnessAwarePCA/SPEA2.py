import itertools
from scipy.spatial.distance import pdist, squareform
import numpy as np

def re(X, Y):
    """calculates the reconstruction error of matrix X with respect to matrix Y.
    in particular it calculates square of the frobenius norm of X - Y """

    return np.linalg.norm(X-Y, ord='fro')**2


def SPEA_2_Cost(theta, coeff, grouped_data, X, sensitive_col):
    """calculates the cost function for the SPEA 2 algorithm. In particular it calculates the
    reconstruction error and a fairness measure"""

    P = coeff[:, theta] @ coeff[:, theta].T  # theta should be the columns to select
    group_combinations = itertools.combinations(range(len(sensitive_col.unique())), 2)
    max_fairness_diff = 0
    for comb in group_combinations:
        group1, group2 = comb[0], comb[1]
        X_1 = grouped_data[group1]
        X_2 = grouped_data[group2]
        approx_X_1 = X_1 @ P
        approx_X_2 = X_2 @ P
        fairness_diff = (re(X_1, approx_X_1) / X_1.shape[0] - re(X_2, approx_X_2) / X_2.shape[
            0]) ** 2
        if fairness_diff > max_fairness_diff:
            max_fairness_diff = fairness_diff
    return [re(X, X @ P) / X.shape[0], max_fairness_diff]


def SPEA_2_fitness(pop_ext_size, pop_ext_cost, ParK):
    """calculates the fitness measure for the SPEA 2 algorithm"""

    # Strength
    pop_ext_SM = np.zeros((pop_ext_size, pop_ext_size))
    pop_ext_S = np.zeros((pop_ext_size, 1))

    for i in range(0, pop_ext_size):
        for j in range(i, pop_ext_size):  # unsure about the i or i +1
            if np.all(pop_ext_cost[i, :] <= pop_ext_cost[j, :]) & np.any(
                    pop_ext_cost[i, :] < pop_ext_cost[j, :]):
                pop_ext_SM[i, j] = 1
            elif np.all(pop_ext_cost[j, :] <= pop_ext_cost[i, :]) & np.any(
                    pop_ext_cost[j, :] < pop_ext_cost[i, :]):
                pop_ext_SM[j, i] = 1
        pop_ext_S[i] = sum(pop_ext_SM[i, :])

    # Raw fitness
    raw_aux = np.tile(pop_ext_S, pop_ext_size)
    raw_aux2 = raw_aux * pop_ext_SM
    pop_ext_r = np.sum(raw_aux2, axis=0)

    f1_ideal, f2_ideal = min(pop_ext_cost[:, 0]), min(pop_ext_cost[:, 1])
    f1_worst, f2_worst = max(pop_ext_cost[:, 0]), max(pop_ext_cost[:, 1])
    f1_norm = ((pop_ext_cost[:, 0] - f1_ideal) / (f1_worst - f1_ideal)).reshape(-1, 1)
    f2_norm = ((pop_ext_cost[:, 1] - f2_ideal) / (f2_worst - f2_ideal)).reshape(-1, 1)
    pop_ext_dist_mat = np.sort(squareform(pdist(np.append(f1_norm, f2_norm, axis=1))),
                               axis=0)  # distance matrix with sorted columns
    pop_ext_dist_mat = np.delete(pop_ext_dist_mat, 0, axis=0)  # delete first row because it is 0

    # density
    pop_ext_dens = 1 / (pop_ext_dist_mat[ParK - 1, :pop_ext_size] + 2)

    # fitness
    pop_ext_fit = pop_ext_r + pop_ext_dens
    return pop_ext_fit


def SPEA_2_selection(ext_size, pop_ext_size, pop_ext_fit, pop_ext, pop_ext_cost):
    """ selects individuals to put in the external set"""
    ext_set = []
    ext_set_cost = []
    ext_set_fit = []
    for i in range(pop_ext_size):
        if pop_ext_fit[i] < 1:
            ext_set += [pop_ext[i, :]]
            ext_set_cost += [pop_ext_cost[i, :]]
            ext_set_fit += [pop_ext_fit[i]]  # 1D list

    ext_set = np.array(ext_set)
    ext_set_cost = np.array(ext_set_cost)
    ext_set_fit = np.array(ext_set_fit)

    f1_ideal, f2_ideal = min(ext_set_cost[:, 0]), min(ext_set_cost[:, 1])
    f1_worst, f2_worst = max(ext_set_cost[:, 0]), max(ext_set_cost[:, 1])

    if ext_set_cost.shape[0] != 1:
        f1_norm = ((ext_set_cost[:, 0] - f1_ideal) / (f1_worst - f1_ideal)).reshape(-1, 1)
        f2_norm = ((ext_set_cost[:, 1] - f2_ideal) / (f2_worst - f2_ideal)).reshape(-1, 1)
    else:  # divide by zero is not possible
        f1_norm = (ext_set_cost[:, 0] - f1_ideal).reshape(-1, 1)
        f2_norm = (ext_set_cost[:, 1] - f2_ideal).reshape(-1, 1)

    ext_dist_mat = squareform(pdist(np.append(f1_norm, f2_norm, axis=1)))  # distance matrix
    ext_dist_mat -= np.eye(len(f1_norm))  # diagonal minus 1

    # analysis of length of external set
    if ext_set.shape[0] > ext_size:  # case with more individuals than the maximum
        while ext_set.shape[0] > ext_size:
            ext_dist_mat_ind = np.argsort(ext_dist_mat.T, axis=0)  # sort rows
            ext_dist_matr2 = np.sort(ext_dist_mat.T,
                                     axis=0)  # cant figure out how to do it with index
            ext_dist_matr2 = np.delete(ext_dist_matr2, 0,
                                       axis=0)  # delete first row, why not column?
            ext_dist_mat_ind = np.delete(ext_dist_mat_ind, 0, axis=0)  # delete first row

            ext_min_ind = np.argmin(ext_dist_matr2[0, :])

            if ext_dist_matr2[1, ext_min_ind] < ext_dist_matr2[
                1, ext_dist_mat_ind[0, ext_min_ind]]:
                ext_set = np.delete(ext_set, ext_min_ind, axis=0)
                ext_set_cost = np.delete(ext_set_cost, ext_min_ind, axis=0)
                ext_set_fit = np.delete(ext_set_fit, ext_min_ind)
                ext_dist_mat = np.delete(ext_dist_mat, ext_min_ind, axis=0)
                ext_dist_mat = np.delete(ext_dist_mat, ext_min_ind, axis=1)
            else:
                ext_set = np.delete(ext_set, ext_dist_mat_ind[0, ext_min_ind], axis=0)
                ext_set_cost = np.delete(ext_set_cost, ext_dist_mat_ind[0, ext_min_ind], axis=0)
                ext_set_fit = np.delete(ext_set_fit, ext_dist_mat_ind[0, ext_min_ind])
                ext_dist_mat = np.delete(ext_dist_mat, ext_dist_mat_ind[0, ext_min_ind], axis=0)
                ext_dist_mat = np.delete(ext_dist_mat, ext_dist_mat_ind[0, ext_min_ind], axis=1)

    elif ext_set.shape[0] < ext_size:  # case with less individuals than maximum
        pop_ext_fit_sort_idx = np.argsort(pop_ext_fit, axis=0)
        ext_set_cost = np.append(ext_set_cost,
                                 pop_ext_cost[pop_ext_fit_sort_idx[ext_set.shape[0]:ext_size], :],
                                 axis=0)
        ext_set_fit = np.append(ext_set_fit,
                                pop_ext_fit[pop_ext_fit_sort_idx[ext_set.shape[0]:ext_size]])
        ext_set = np.append(ext_set, pop_ext[pop_ext_fit_sort_idx[ext_set.shape[0]:ext_size], :],
                            axis=0)

    return ext_set, ext_set_cost, ext_set_fit


def SPEA_2_mating_pool(mut_max, cros_max, ext_set, ext_set_size, pop_size, ext_set_fit, iteration,
                       gen_numb, ell, feat_num):
    """ function for mating and crossover for the population for SPEA 2 algorithm"""
    pop_aux1 = np.zeros((pop_size, ell))
    MP1 = np.random.randint(0, ext_set_size, size=pop_size)
    MP2 = np.random.randint(0, ext_set_size, size=pop_size)

    ext_set_fit_sort_idx = np.argsort(ext_set_fit)

    for i in range(pop_size):
        if MP1[i] <= MP2[i]:
            pop_aux1[i, :] = ext_set[ext_set_fit_sort_idx[MP1[i]], :]
        else:
            pop_aux1[i, :] = ext_set[ext_set_fit_sort_idx[MP2[i]], :]

    # Crossover / Recombination
    pop_aux2 = np.zeros((pop_size, ell))
    cross_aux_perm = np.random.permutation(cros_max)

    for i in range(round(cros_max / 2)):
        inter = np.intersect1d(pop_aux1[cross_aux_perm[1], :], pop_aux1[cross_aux_perm[0], :])

        if len(inter) >= (ell - 1):
            pop_aux2[max(0, 2 * i - 1), :] = pop_aux1[cross_aux_perm[0], :]
            pop_aux2[2 * i, :] = pop_aux1[cross_aux_perm[1], :]
        else:
            pop_aux2[max(0, 2 * i - 1), :len(inter)] = inter
            pop_aux2[2 * i, :len(inter)] = inter
            p1_aux = np.setdiff1d(pop_aux1[cross_aux_perm[0], :], inter)
            p2_aux = np.setdiff1d(pop_aux1[cross_aux_perm[1], :], inter)
            prob = np.random.permutation(len(p1_aux))
            prob2 = np.random.randint(0, round(
                (iteration - 1) * (1 - (len(p1_aux) - 1)) / (gen_numb - 1) + (len(p1_aux) - 1)))
            pop_aux2[max(2 * i - 1, 0), len(inter):len(inter) + prob2] = p2_aux[prob[:prob2]]
            pop_aux2[max(2 * i - 1, 0), len(inter) + prob2:] = p1_aux[prob[prob2:]]
            pop_aux2[2 * i, len(inter):len(inter) + prob2] = p1_aux[prob[:prob2]]
            pop_aux2[2 * i, len(inter) + prob2:] = p2_aux[prob[prob2:]]
        cross_aux_perm = np.delete(cross_aux_perm, [0, 1], axis=0)

    # Mutation
    mut_aux_perm = np.random.permutation(mut_max) + cros_max
    for i in range(round(cros_max / 2) * 2, pop_size):
        prob = np.random.permutation(ell)
        prob2 = np.random.randint(0, round((iteration - 1) * (1 - (ell)) / (gen_numb - 1) + (ell)))
        p_aux = np.setdiff1d(np.arange(feat_num), pop_aux1[mut_aux_perm[0], prob[prob2:]])
        prob3 = np.random.permutation(len(p_aux))
        pop_aux2[i, prob[:prob2]] = p_aux[prob3[:prob2]]
        pop_aux2[i, prob[prob2:]] = pop_aux1[mut_aux_perm[0], prob[prob2:]]
        mut_aux_perm = np.delete(mut_aux_perm, 0, axis=0)
    return pop_aux2