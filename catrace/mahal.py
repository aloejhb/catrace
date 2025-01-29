import numpy as np
import pandas as pd
import seaborn as sns
from scipy.spatial.distance import mahalanobis, euclidean
import catrace.process_time_trace as ptt
from catrace.utils import get_seed_from_hash

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


def compute_distances_df(df, window=None, metric='mahal', reg=0,
                         do_shuffle_manifold_pair_labels=False, shuffle_seed_value=None):
    df = ptt.select_time_points(df, window)
    odor_list = list(df.index.unique('odor'))
    num_odors = len(odor_list)

    if do_shuffle_manifold_pair_labels:
        if shuffle_seed_value is not None:
            master_rng = np.random.default_rng(shuffle_seed_value)
            seed_values = master_rng.integers(0, 1e9, size=num_odors*num_odors).tolist()
        else:
            seed_values = [None] * num_odors*num_odors

    distances_dict = dict()
    for odor1 in odor_list:
        # data1 = df.xs(odor1, level='odor')
        manifold1 = df.xs(odor1, level='odor')
        for odor2 in odor_list:
            manifold2 = df.xs(odor2, level='odor')

            if do_shuffle_manifold_pair_labels:
                manifold1, manifold2 = shuffle_manifold_pair_labels(manifold1, manifold2, seed_value=seed_values.pop(0))

            center2 = manifold2.mean(axis=0)

            if metric == 'mahal':
                inv_cov_mat2 = invert_cov_mat(np.cov(manifold2.transpose()), reg=reg)
                distances = compute_mahals(manifold1.to_numpy(),
                                          center2.to_numpy(),
                                          inv_cov_mat2)
            elif metric == 'euclidean':
                distances = compute_euclideans(manifold1.to_numpy(),
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

def compute_center_euclidean(manifold1, manifold2):
    """
    Compute euclidean distance between the centers of two manifolds
    Args:
        manifold1: pd.DataFrame. Rows are time points and columns are neurons.
        manifold2: pd.DataFrame. Rows are time points and columns are neurons.
    Returns:
        dist: float. Euclidean distance between the centers of the two manifolds.
    """
    center1 = manifold1.mean(axis=0) # Average over time points
    center2 = manifold2.mean(axis=0)
    dist = euclidean(center1, center2)

    return dist

def shuffle_manifold_pair_labels(manifold1, manifold2, seed_value):
    random_state = np.random.default_rng(seed_value)
    # Concatenate manifold1 and manifold2
    manifold1and2 = pd.concat([manifold1, manifold2])
    # Shuffle the rows
    manifold1and2_shuffled = manifold1and2.sample(frac=1, random_state=random_state)
    # Set the index to the original index
    manifold1and2_shuffled.index = manifold1and2.index
    # Split the shuffled manifold into two
    manifold1 = manifold1and2_shuffled.iloc[:len(manifold1)]
    manifold2 = manifold1and2_shuffled.iloc[len(manifold1):]
    return manifold1, manifold2

def compute_center_euclidean_distance_mat(df, odor_list, window, do_shuffle_manifold_pair_labels=False, shuffle_seed_value=None):
    df = ptt.select_time_points(df, window)
    df = ptt.select_odors_and_sort(df, odor_list)

    # Compute euclidean distance between centers
    dist_mat = pd.DataFrame(index=odor_list, columns=odor_list, dtype=float)
    # The level name is 'odor' for the index and 'ref_odor' for the columns
    dist_mat.index.name = 'odor'
    dist_mat.columns.name = 'ref_odor'

    if do_shuffle_manifold_pair_labels:
        if shuffle_seed_value is not None:
            master_rng = np.random.default_rng(shuffle_seed_value)
            seed_values = master_rng.integers(0, 1e9, size=len(odor_list)*len(odor_list)).tolist()
        else:
            seed_values = [None] * len(odor_list)*len(odor_list)

    for odor1 in odor_list:
        for odor2 in odor_list:
            manifold1 = df.xs(odor1, level='odor')
            manifold2 = df.xs(odor2, level='odor')
            if do_shuffle_manifold_pair_labels:
                manifold1, manifold2 = shuffle_manifold_pair_labels(manifold1, manifold2, seed_value=seed_values.pop(0))
                if shuffle_seed_value is not None:
                    shuffle_seed_value = shuffle_seed_value + 1
            dist = compute_center_euclidean(manifold1, manifold2)
            dist_mat.loc[odor1, odor2] = dist

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
