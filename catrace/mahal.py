import numpy as np
import pandas as pd
import seaborn as sns
from scipy.spatial.distance import mahalanobis, euclidean

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


def compute_distances_df(df, window=None, metric='mahal', reg=0):
    if window is not None:
        times = df.index.get_level_values('time')
        idxs = (times >= window[0]) &(times <= window[1])
        df = df.loc[idxs, :]

    odor_list = list(df.index.unique('odor'))

    # Compute center for each odor
    centers = df.groupby(level='odor', sort=False).mean()

    if metric == 'mahal':
        # Compute cov_mat for each odor
        cov_mats = dict()
        for name, group in df.groupby(level='odor', sort=False):
            cov_mats[name] = np.cov(group.transpose())

        # Compute inv_cov_mat for each odor
        inv_cov_mats = dict()
        for key, val in cov_mats.items():
            inv_cov_mats[key] = invert_cov_mat(val, reg=reg)

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


def compute_mahals_mat(df):
    mahals_df = compute_mahals_df(df)
    mahals_mat = get_mean_mahals_mat(mahals_df)
    return mahals_mat


def get_mean_mahals_mat(mahals_df, odor_list):
    mahals_mean = mahals_df.mean(axis=1)
    mahals_mean.name = 'mean_mahal_dist'
    mahals_mean = mahals_mean.reset_index()


    mahals_mean['order'] = range(len(mahals_mean))
    mahals_mat = mahals_mean.pivot_table(index='odor', columns='ref_odor', values='mean_mahal_dist')

    mahals_mat.index = pd.Categorical(mahals_mat.index, categories=odor_list, ordered=True)
    mahals_mat = mahals_mat.sort_index()
    mahals_mat = mahals_mat[odor_list]
    return mahals_mat


def plot_mean_mahals_mat(mahals_mat, ax=None, vmin=None, vmax=None):
    ax = sns.heatmap(mahals_mat, ax=ax, cmap="RdBu_r", cbar=False,
                      vmin=vmin, vmax=vmax,
                      xticklabels=mahals_mat.columns, yticklabels=mahals_mat.index)
    return ax


def plot_mahals_mat_df(df, ax=None, odor_list=None, **kwargs):
    mahals_mat = get_mean_mahals_mat(df, odor_list=odor_list)
    ax = plot_mean_mahals_mat(mahals_mat, ax=ax, **kwargs)
    return ax
