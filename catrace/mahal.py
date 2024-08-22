import numpy as np
import pandas as pd
import seaborn as sns
from scipy.spatial.distance import mahalanobis, euclidean
import catrace.process_time_trace as ptt

def invert_cov_mat(cov_mat, reg=1e-5):
    reg_term = reg * np.identity(cov_mat.shape[0])
    cov_mat += reg_term
    # Matrix Inversion using SVD for numerical stability
    u, s, v = np.linalg.svd(cov_mat)
    inv_cov_mat = np.dot(v.T, np.dot(np.diag(1/s), u.T))
    return inv_cov_mat

def compute_mahals(points, ref, inv_cov_mat):
    mahals = [mahalanobis(p, ref, inv_cov_mat) for p in points]
    return mahals

def compute_euclideans(points, ref):
    mahals = [euclidean(p, ref) for p in points]
    return mahals


def compute_distances_df(df, window=None, model_window=None, model_trials=None, metric='mahal', reg=0):
    df = ptt.select_time_points(df, window)
    odor_list = list(df.index.unique('odor'))

    model_df = df
    if model_window is not None:
        model_df = ptt.select_time_points(df, model_window)

    if model_trials is not None:
        idx = model_df.index.get_level_values('trial').isin(model_trials)
        model_df = model_df[idx]

    # Compute center for each odor
    centers = model_df.groupby(level='odor', sort=False).mean()

    if metric == 'mahal':
        # Compute cov_mat for each odor
        cov_mats = dict()
        for name, group in model_df.groupby(level='odor', sort=False, observed=True):
            cov_mats[name] = np.cov(group.transpose())

        # Compute inv_cov_mat for each odor
        inv_cov_mats = dict()
        for key, val in cov_mats.items():
            try:
                inv_cov_mats[key] = invert_cov_mat(val, reg=reg)
            except:
                import pdb; pdb.set_trace()

    distances_dict = dict()
    for odor1 in odor_list:
        data1 = df.xs(odor1, level='odor')
        for odor2 in odor_list:
            center2 = centers.loc[odor2]

            if metric == 'mahal':
                inv_cov_mat2 = inv_cov_mats[odor2]
                distances = compute_mahals(data1.to_numpy(),
                                          center2.to_numpy(),
                                          inv_cov_mat2)
            elif metric == 'euclidean':
                distances = compute_euclideans(data1.to_numpy(),
                                              center2.to_numpy())
            else:
                raise ValueError('Metric should be either mahal or euclidean.')

            distances_dict[(odor1, odor2)] = distances

    # Convert to DataFrame with separate columns for odors
    multi_index = pd.MultiIndex.from_tuples(distances_dict.keys(), names=['odor', 'ref_odor'])

    # Convert to DataFrame using MultiIndex
    distances_df = pd.DataFrame(list(distances_dict.values()), index=multi_index)
    return distances_df


def sample_neuron_and_select_odors(df, sample_size, seed=None, odor_list=None):
    df = df.dropna()
    df = ptt.sample_neuron(df, sample_size=sample_size, random_state=seed)
    if odor_list is not None:
        df = ptt.select_odors_df(df, odor_list)
        df = ptt.sort_odors(df, odor_list)
    return df


def compute_distances_mat(df, odor_list, **kwargs):
    dist_df = compute_distances_df(df, **kwargs)
    dist_mat = get_mean_dist_mat(dist_df, odor_list)
    return dist_mat


def sample_neuron_and_comopute_distance_mat(df, sample_size, seed=None, **kwargs):
    df = ptt.sample_neuron(df, sample_size=sample_size, seed=seed)
    dist_mat = compute_distances_mat(df, **kwargs)
    return dist_mat


def get_mean_dist_mat(dist_df, odor_list):
    dist_mean = dist_df.mean(axis=1)
    dist_mean.name = 'mean_dist'
    dist_mean = dist_mean.reset_index()


    dist_mean['order'] = range(len(dist_mean))
    dist_mat = dist_mean.pivot_table(index='odor',
                                     columns='ref_odor',
                                     values='mean_dist')

    dist_mat.index = pd.Categorical(dist_mat.index,
                                    categories=odor_list,
                                    ordered=True)
    dist_mat = dist_mat.sort_index()
    dist_mat = dist_mat[odor_list]
    return dist_mat


def plot_mean_dist_mat(dist_mat, ax=None, vmin=None, vmax=None):
    ax = sns.heatmap(dist_mat, ax=ax, cmap="RdBu_r", cbar=False,
                      vmin=vmin, vmax=vmax,
                      xticklabels=dist_mat.columns, yticklabels=dist_mat.index)
    return ax


def plot_dist_mat_df(df, ax=None, odor_list=None, **kwargs):
    dist_mat = get_mean_dist_mat(df, odor_list=odor_list)
    ax = plot_mean_dist_mat(dist_mat, ax=ax, **kwargs)
    return ax


def select_odors_mat(mat, odor_list):
    return mat.loc[odor_list][odor_list]
